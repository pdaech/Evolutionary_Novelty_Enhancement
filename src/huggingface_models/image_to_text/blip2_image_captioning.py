import torch
from PIL import Image
from transformers import Blip2ForConditionalGeneration, AutoProcessor

from src.huggingface_models.base_strategy import CaptionModelStrategy


class Blip2CaptioningModel(CaptionModelStrategy):

    def __init__(
        self,
        device: torch.device,
        dtype: torch.dtype,
        cache_dir: str,
        prompt: str = "",
        model: str = "Salesforce/blip2-opt-2.7b",
    ) -> None:
        self.device = device
        self.dtype = dtype
        self.prompt = prompt
        self.model = Blip2ForConditionalGeneration.from_pretrained(
            model,
            torch_dtype=dtype,
            cache_dir=cache_dir,
            use_safetensors=True,
            device_map="auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(model)

    def caption(self, pixel_image: Image.Image) -> str:

        inputs = self.processor(
            images=pixel_image, text=self.prompt, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=60, num_beams=5, early_stopping=True
            )

        caption = self.processor.decode(generated_ids[0], skip_special_tokens=True)
        return caption

    def caption_batch(self, pixel_images: list[Image.Image]) -> list[str]:
        inputs = self.processor(
            images=pixel_images,
            text=[self.prompt] * len(pixel_images),
            return_tensors="pt",
            padding=True,
        ).to(self.device)

        with torch.no_grad():
            generated_ids = self.model.generate(
                **inputs, max_new_tokens=60, num_beams=5, early_stopping=True
            )

        captions = [
            self.processor.decode(generated_ids[i], skip_special_tokens=True)
            for i in range(generated_ids.size(0))
        ]
        return captions
