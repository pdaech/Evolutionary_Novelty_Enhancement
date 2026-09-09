import logging
from threading import Lock
from typing import ClassVar

import torch

from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel
from src.huggingface_models.image_embedding.clip_embedding import ClipEmbeddingModel
from src.huggingface_models.text_to_image.stable_diffusion_xl import (
    StableDiffusionXLModel,
)
from src.sdxl_options import (
    DEFAULT_SDXL_GUIDANCE_SCALE,
    DEFAULT_SDXL_NUM_INFERENCE_STEPS,
)

logger = logging.getLogger(__name__)


class ModelLoader:
    _instances: ClassVar[dict] = {}
    _lock: Lock = Lock()

    def __new__(cls, cache_dir: str):
        with cls._lock:
            if cls not in cls._instances:
                instance = super().__new__(cls)
                cls._instances[cls] = instance
        return cls._instances[cls]

    def __init__(self, cache_dir: str):

        if hasattr(self, "_initialized") and self._initialized:
            if cache_dir != self.cache_dir:
                raise ValueError(
                    "ModelLoader is already initialized with a different cache_dir"
                )
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.cache_dir = cache_dir
        self.sdxl = None
        self.blip2_embeddings = None
        self.clip_embeddings = None
        self._initialized = True

    def load_sdxl(
        self,
        revision: str | None = None,
        num_inference_steps: int = DEFAULT_SDXL_NUM_INFERENCE_STEPS,
        guidance_scale: float = DEFAULT_SDXL_GUIDANCE_SCALE,
    ) -> StableDiffusionXLModel:
        if self.sdxl is None:
            self.sdxl = StableDiffusionXLModel(
                self.device,
                self.dtype,
                self.cache_dir,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                revision=revision,
            )
        elif (
            self.sdxl.requested_revision != revision
            or self.sdxl.num_inference_steps != num_inference_steps
            or self.sdxl.guidance_scale != guidance_scale
        ):
            raise ValueError(
                "SDXL is already loaded with a different requested configuration"
            )

        return self.sdxl

    def load_sdxl_turbo(self):
        pass

    def load_clip_embeddings(self) -> ClipEmbeddingModel:
        if self.clip_embeddings is None:
            self.clip_embeddings = ClipEmbeddingModel(
                self.device, self.dtype, self.cache_dir
            )
        return self.clip_embeddings

    def load_blip2_embeddings(self) -> Blip2EmbeddingModel:
        if self.blip2_embeddings is None:
            self.blip2_embeddings = Blip2EmbeddingModel(
                self.device, self.dtype, self.cache_dir
            )
        return self.blip2_embeddings

    def load_blip2_captions(self):
        pass
