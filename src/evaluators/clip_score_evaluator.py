import torch
from PIL import Image
from transformers import CLIPProcessor, CLIPModel


class CLIPScoreEvaluator:

    def __init__(
        self,
        prompt: str,
        prefix: str = "A photo depicts ",
        device: str = "cuda",
        model_id: str = "openai/clip-vit-base-patch32",
        use_prefix: bool = True,
    ) -> None:

        self.device = device

        self.model = CLIPModel.from_pretrained(model_id).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_id)

        if use_prefix and not prompt.startswith(prefix):
            prompt = f"A photo depicts {prompt}"

        text_inputs = self.processor(
            text=[prompt], return_tensors="pt", padding=True
        ).to(self.device)
        with torch.no_grad():
            text_features = self.model.get_text_features(**text_inputs)
            self.text_features = text_features / text_features.norm(
                p=2, dim=-1, keepdim=True
            )

    def evaluate_batch(self, images: list[Image.Image]) -> list[float]:

        images = [img.convert("RGB") for img in images]

        image_inputs = self.processor(images=images, return_tensors="pt").to(
            self.device
        )

        with torch.no_grad():

            image_features = self.model.get_image_features(**image_inputs)
            image_features = image_features / image_features.norm(
                p=2, dim=-1, keepdim=True
            )

            similarities = (self.text_features @ image_features.T).squeeze(0)

            w = 2.5
            clip_scores = w * torch.clamp(similarities, min=0.0)

        return clip_scores.cpu().tolist()
