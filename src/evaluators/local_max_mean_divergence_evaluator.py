import os
from pathlib import Path
from typing import List, Any
from torch.nn import functional as F
import torch
from collections import deque

from src.evaluators.global_max_mean_divergence_evaluator import (
    GlobalMaxMeanDivergenceEvaluator,
)
from src.evaluators.base_evaluator import Evaluator
from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel


class LocalMaxMeanDivergenceEvaluator(GlobalMaxMeanDivergenceEvaluator):

    def __init__(
        self,
        prompt,
        device: str,
        archive_size: int = 200,
        archive_threshold: float = 0.1,
    ):
        self.device = device
        self.name = "LocalMaxMeanDivergence"

        self.archive_size = archive_size
        self.archive_threshold = archive_threshold

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
        print(
            f"Calculated scores for batch with size {len(image_features)} and Shape: {image_features[0].shape}"
        )
        return [{"name": self.name, "score": score} for score in scores]

    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel

    def update(self, population: list[torch.Tensor]) -> None:

        combined_stack = torch.cat(population, dim=0)

        mean = torch.mean(combined_stack, dim=0, keepdim=True)
        print(f"Mean Tensor Updated, with Shape: {mean.shape}")
        self.mean_embedding = mean


class LocalArchivMaxMeanDivergenceEvaluator(GlobalMaxMeanDivergenceEvaluator):

    def __init__(
        self,
        prompt,
        device: str,
        archive_size: int = 200,
        archive_threshold: float = 0.1,
    ):
        self.device = device
        self.name = "LocalArchivMaxMeanDivergence"

        self.archive_size = archive_size
        self.archive_threshold = archive_threshold
        self.archive = deque(maxlen=archive_size)

    def evaluate(self, image_features: torch.Tensor, *args, **kwargs) -> dict[str, Any]:

        cos = F.cosine_similarity(image_features, self.mean_embedding)
        similarity = cos.item()
        normalised_similarity = (similarity + 1) / 2
        inversed_similarity: float = 1 - normalised_similarity

        if inversed_similarity > self.archive_threshold:
            self.archive.append(image_features.detach().clone())

        return {"name": self.name, "score": inversed_similarity}

    def evaluate_batch(
        self, image_features: List[torch.Tensor], *args, **kwargs
    ) -> list[dict[str, Any]]:

        stack = torch.cat(image_features, dim=0)

        stack = stack.to(self.device)
        cos_similarities = F.cosine_similarity(stack, self.mean_embedding)
        normalised_similarities = (cos_similarities + 1) / 2
        inverse_similarities = 1 - normalised_similarities

        high_score_indices = (inverse_similarities > self.archive_threshold).nonzero(
            as_tuple=True
        )[0]

        for idx in high_score_indices:
            feature = stack[idx].unsqueeze(0).detach().clone()
            self.archive.append(feature)

        scores = inverse_similarities.tolist()
        print(
            f"Calculated scores for batch with size {len(image_features)} and Shape: {image_features[0].shape}"
        )
        return [{"name": self.name, "score": score} for score in scores]

    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel

    def update(self, population: list[torch.Tensor]) -> None:

        current_pop_stack = torch.cat(population, dim=0)

        if len(self.archive) > 0:

            archive_stack = torch.cat(list(self.archive), dim=0).to(self.device)

            combined_stack = torch.cat([current_pop_stack, archive_stack], dim=0)
            print(
                f"Updating mean with Population ({len(population)}) + Archive ({len(self.archive)})"
            )
        else:
            combined_stack = current_pop_stack
            print(
                f"Updating mean with Population ({len(population)}) only (Archive empty)"
            )

        mean = torch.mean(combined_stack, dim=0, keepdim=True)
        print(f"Mean Tensor Updated, with Shape: {mean.shape}")
        self.mean_embedding = mean
