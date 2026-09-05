from importlib.metadata import PackageNotFoundError, version

import PIL
import torch
from diffusers import StableDiffusionXLPipeline

from src.huggingface_models.base_strategy import GenerativModelStrategy


class StableDiffusionXLModel(GenerativModelStrategy):
    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        cache_dir: str,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
        compile_pipeline: bool = False,
        model: str = "stabilityai/stable-diffusion-xl-base-1.0",
    ) -> None:

        self.num_inference_steps = num_inference_steps
        self.guidance_scale = guidance_scale
        self.model_id = model
        self.dtype = str(dtype)
        self.device = str(device)
        self.model = StableDiffusionXLPipeline.from_pretrained(
            model,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            use_safetensors=True,
        )
        self.model.set_progress_bar_config(leave=False)
        self.model.set_progress_bar_config(disable=True)
        self.model.to(device=device)
        if compile_pipeline:
            self.model.unet = torch.compile(
                self.model.unet, mode="reduce-overhead", fullgraph=True
            )

    def config_metadata(self) -> dict:
        pipeline_config = getattr(self.model, "config", None)
        resolved_revision = getattr(pipeline_config, "_commit_hash", None)
        return {
            "model_id": self.model_id,
            "resolved_revision": (
                str(resolved_revision) if resolved_revision is not None else None
            ),
            "dtype": self.dtype,
            "device": self.device,
            "num_inference_steps": self.num_inference_steps,
            "guidance_scale": self.guidance_scale,
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
        num_inference_steps: int = 50,
        guidance_scale: float = 7.0,
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
