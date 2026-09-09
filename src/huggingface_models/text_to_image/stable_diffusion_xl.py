from importlib.metadata import PackageNotFoundError, version

import PIL
import torch
from diffusers import StableDiffusionXLPipeline

from src.huggingface_models.base_strategy import GenerativModelStrategy
from src.model_revisions import load_pinned_pipeline
from src.sdxl_options import (
    DEFAULT_SDXL_GUIDANCE_SCALE,
    DEFAULT_SDXL_NUM_INFERENCE_STEPS,
    EXPECTED_SDXL_SCHEDULER_CLASS,
)


class StableDiffusionXLModel(GenerativModelStrategy):
    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        cache_dir: str,
        num_inference_steps: int = DEFAULT_SDXL_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_SDXL_GUIDANCE_SCALE,
        compile_pipeline: bool = False,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        revision: str | None = None,
    ) -> None:

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.model_id = model
        self.dtype = str(dtype)
        self.device = str(device)
        self.requested_revision = revision
        if revision is not None:
            self.model, self._revision_metadata = load_pinned_pipeline(
                StableDiffusionXLPipeline,
                model_id=model,
                revision=revision,
                cache_dir=cache_dir,
                dtype=dtype,
            )
        else:
            # Preserve unpinned loading for existing library/refiner callers.
            self.model = StableDiffusionXLPipeline.from_pretrained(
                model,
                torch_dtype=dtype,
                cache_dir=cache_dir,
                use_safetensors=True,
            )
            pipeline_config = getattr(self.model, "config", None)
            commit = getattr(pipeline_config, "_commit_hash", None)
            self._revision_metadata = {
                "requested_revision": None,
                "resolved_revision": str(commit) if commit is not None else None,
                "revision_source": "pipeline_config" if commit is not None else None,
                "snapshot_path": None,
            }
        scheduler = getattr(self.model, "scheduler", None)
        if scheduler is None:
            raise RuntimeError("Loaded SDXL pipeline has no scheduler")
        self.scheduler_class = type(scheduler).__name__
        if self.scheduler_class != EXPECTED_SDXL_SCHEDULER_CLASS:
            raise RuntimeError(
                "Thesis-compatible SDXL generation requires "
                f"{EXPECTED_SDXL_SCHEDULER_CLASS}, got {self.scheduler_class}"
            )
        scheduler_config = getattr(scheduler, "config", {})
        scheduler_keys = (
            "beta_end",
            "beta_schedule",
            "beta_start",
            "prediction_type",
            "timestep_spacing",
            "use_karras_sigmas",
        )
        self.scheduler_config = {
            key: scheduler_config[key]
            for key in scheduler_keys
            if key in scheduler_config
        }
        self.model.set_progress_bar_config(leave=False)
        self.model.set_progress_bar_config(disable=True)
        self.model.to(device=device)
        if compile_pipeline:
            self.model.unet = torch.compile(
                self.model.unet, mode="reduce-overhead", fullgraph=True
            )

    def config_metadata(self) -> dict:
        return {
            "model_id": self.model_id,
            **self._revision_metadata,
            "dtype": self.dtype,
            "device": self.device,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
            "scheduler_class": self.scheduler_class,
            "scheduler_config": self.scheduler_config,
            "software": {
                "torch": str(torch.__version__),
                "diffusers": _package_version("diffusers"),
                "pillow": str(PIL.__version__),
            },
        }

    def generate(self, noise_emds: torch.Tensor, prompt: str):

        image = self.model(
            prompt=prompt,
            latents=noise_emds,
            output_type="pil",
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            disable_tqdm=True,
        ).images[0]

        return image

    def generate_batch(self, noise_emds: list[torch.Tensor], prompt: str):

        stack = torch.cat(noise_emds, dim=0)
        images = self.model(
            prompt=[prompt] * len(noise_emds),
            latents=stack,
            output_type="pil",
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
            disable_tqdm=True,
        ).images

        return images


class StableDiffusionXLRefinerStrategy(StableDiffusionXLModel):
    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cache_dir: str,
        compile_pipeline: bool = False,
        num_inference_steps: int = DEFAULT_SDXL_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_SDXL_GUIDANCE_SCALE,
        high_noise_frac=0.8,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
        refiner: str = "stabilityai/stable-diffusion-xl-refiner-1.0",
    ):
        super().__init__(
            device,
            dtype,
            cache_dir,
            num_inference_steps,
            guidance_scale,
            compile_pipeline,
            model,
        )
        self.high_noise_frac = high_noise_frac
        self.refiner = StableDiffusionXLPipeline.from_pretrained(
            refiner,
            text_encoder_2=self.model.text_encoder_2,
            vae=self.model.vae,
            torch_dtype=dtype,
            use_safetensors=True,
            cache_dir=cache_dir,
        )
        self.refiner.to(device=device)
        if compile_pipeline:
            self.refiner.unet = torch.compile(
                self.refiner.unet, mode="reduce-overhead", fullgraph=True
            )

    def generate(self, noise_emds: torch.Tensor, prompt: str):
        image = self.model(
            prompt=prompt,
            latents=noise_emds,
            output_type="latent",
            denoising_end=self.high_noise_frac,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images
        image = self.refiner(
            prompt=prompt,
            num_inference_steps=self.num_inference_steps,
            denoising_start=self.high_noise_frac,
            image=image,
        ).images[0]
        return image

    def generate_batch(self, noise_emds: list[torch.Tensor], prompt: str):
        stack = torch.cat(noise_emds, dim=0)
        images = self.model(
            prompt=[prompt] * len(noise_emds),
            latents=stack,
            output_type="latent",
            denoising_end=self.high_noise_frac,
            num_inference_steps=self.num_inference_steps,
            guidance_scale=self.guidance_scale,
        ).images

        image = self.refiner(
            prompt=[prompt] * len(noise_emds),
            num_inference_steps=self.num_inference_steps,
            denoising_start=self.high_noise_frac,
            image=images,
        ).images
        return image


def _package_version(name: str) -> str | None:
    try:
        return version(name)
    except PackageNotFoundError:
        return None
