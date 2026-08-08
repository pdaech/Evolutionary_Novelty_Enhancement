import torch
from PIL import Image

from src.huggingface_models.base_strategy import EmbeddingModelStrategy
from transformers import Blip2Model, AutoProcessor


class Blip2EmbeddingModel(EmbeddingModelStrategy):

    def __init__(
        self,
        device: str,
        dtype: torch.dtype,
        cache_dir: str,
        model: str = "Salesforce/blip2-opt-2.7b",
    ) -> None:

        self.device = device
        self.model = Blip2Model.from_pretrained(
            model, torch_dtype=dtype, cache_dir=cache_dir, use_safetensors=True
        )
        self.model.to(self.device)
        self.processor = AutoProcessor.from_pretrained(model)

    def image_features_extraction(self, pixel_image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pixel_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs).pooler_output

            return image_embeds

    def batch_image_features_extraction(
        self, pixel_images: list[Image.Image]
    ) -> list[torch.Tensor]:

        inputs = self.processor(images=pixel_images, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            image_embeds = self.model.get_image_features(**inputs).pooler_output

        output = [image_embeds[i].unsqueeze(0) for i in range(len(pixel_images))]
        return output

    def qformer_feature_extraction(self, pixel_image: Image.Image) -> torch.Tensor:
        inputs = self.processor(images=pixel_image, return_tensors="pt").to(self.device)

        with torch.no_grad():
            image_embeds = self.model.get_qformer_features(
                **inputs, legacy_output=False
            )

            return image_embeds

    def batch_qformer_feature_extraction(
        self, pixel_images: list[Image.Image]
    ) -> list[torch.Tensor]:

        inputs = self.processor(images=pixel_images, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():
            image_embeds = self.model.get_qformer_features(
                **inputs, legacy_output=False
            )

        output = [image_embeds[i] for i in range(len(pixel_images))]
        return output
