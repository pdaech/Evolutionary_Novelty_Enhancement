import torch
from PIL import Image

from src.huggingface_models.base_strategy import EmbeddingModelStrategy
from transformers import CLIPModel, CLIPProcessor


class ClipEmbeddingModel(EmbeddingModelStrategy):

    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        cache_dir: str,
        model: str = "openai/clip-vit-base-patch32",
    ) -> None:

        self.device = device
        self.model = CLIPModel.from_pretrained(
            model,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            use_safetensors=True,
            device_map="auto",
        )

        self.model.to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model)

    def image_features_extraction(self, pixel_image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pixel_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)
            return image_embeds

    def batch_image_features_extraction(
        self, pixel_images: list[Image.Image]
    ) -> list[torch.Tensor]:

        inputs = self.processor(images=pixel_images, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs)

        image_embeds = [image_embeds[i] for i in range(len(pixel_images))]
        return image_embeds
