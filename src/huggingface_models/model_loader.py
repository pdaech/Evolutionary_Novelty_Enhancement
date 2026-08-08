import logging
import os
from dataclasses import dataclass
from threading import Thread, Lock

import torch

from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel
from src.huggingface_models.image_embedding.clip_embedding import ClipEmbeddingModel
from src.huggingface_models.image_to_text.blip2_image_captioning import (
    Blip2CaptioningModel,
)
from src.huggingface_models.text_to_image.stable_diffusion_xl import (
    StableDiffusionXLRefinerStrategy,
    StableDiffusionXLModel,
)

logger = logging.getLogger(__name__)


class ModelLoader(object):
    _instances = {}
    _lock: Lock = Lock()

    def __new__(cls, cache_dir: str):
        with cls._lock:
            if cls not in cls._instances:
                instance = super(ModelLoader, cls).__new__(cls)
                cls._instances[cls] = instance
        return cls._instances[cls]

    def __init__(self, cache_dir: str):

        if hasattr(self, "_initialized") and self._initialized:
            return
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        self.cache_dir = cache_dir
        self.sdxl = None
        self.blip2_embeddings = None
        self.clip_embeddings = None

    def load_sdxl(self) -> StableDiffusionXLModel:
        if self.sdxl is None:
            self.sdxl = StableDiffusionXLModel(self.device, self.dtype, self.cache_dir)

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
