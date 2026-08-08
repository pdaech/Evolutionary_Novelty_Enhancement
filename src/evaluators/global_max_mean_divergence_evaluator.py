import os
from pathlib import Path
from typing import List, Any
from torch.nn import functional as F
import torch

from src.evaluators.base_evaluator import Evaluator
from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel


class GlobalMaxMeanDivergenceEvaluator(Evaluator):

    def __init__(self, prompt, device: str):
        base_path = Path(os.environ.get("BASE_PATH", ""))
        path = Path(os.environ.get("BLIP_2_MEAN", ""))
        file = base_path / path / f"{prompt.replace(' ', '-')}.pt"
        self.mean_embedding = torch.load(file)
        self.mean_embedding = self.mean_embedding.to(device)
        self.device = device
        self.name = "GlobalMaxMeanDivergence"

    def evaluate(self, image_features: torch.Tensor, *args, **kwargs) -> dict[str, Any]:

        cos = F.cosine_similarity(image_features, self.mean_embedding)
        similarity = cos.item()
        normalised_similarity = (similarity + 1) / 2
        inversed_similarity: float = 1 - normalised_similarity
        return {"name": self.name, "score": inversed_similarity}

    def evaluate_batch(
        self, image_features: List[torch.Tensor], *args, **kwargs
    ) -> list[dict[str, Any]]:

        stack = torch.cat(image_features, dim=0)

        stack = stack.to(self.device)
        cos_similarities = F.cosine_similarity(stack, self.mean_embedding)
        normalised_similarities = (cos_similarities + 1) / 2
        inverse_similarities = 1 - normalised_similarities
        scores = inverse_similarities.tolist()
        return [{"name": self.name, "score": score} for score in scores]

    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel
