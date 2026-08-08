import os
from pathlib import Path
from typing import List, Any
from torch.nn import functional as F
import torch

from src.evaluators.global_max_mean_divergence_evaluator import (
    GlobalMaxMeanDivergenceEvaluator,
)
from src.evaluators.base_evaluator import Evaluator
from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel
import numpy as np
from sklearn.neighbors import KernelDensity


class LocalKernelDensityEvaluator(GlobalMaxMeanDivergenceEvaluator):

    def __init__(
        self,
        prompt,
        device: str,
        kernel: str = "gaussian",
        metric: str = "euclidean",
        bandwidth: float | str = 0.35,
    ):
        self.device = device
        self.name = "LocalKernelDensity"
        self.kernel = kernel
        self.metric = metric

        if type(bandwidth) == str:
            assert bandwidth in [
                "scott",
                "silverman",
            ], "Specified rule is not available for selection"
        if type(bandwidth) == float:
            assert bandwidth >= 0.0, "Bandwidth must be greater than or equal to 1.0"
        self.rule = bandwidth

    def evaluate(self, image_features: torch.Tensor, *args, **kwargs) -> dict[str, Any]:

        np_image_features = image_features.cpu().numpy()
        log_density = self.kde.score_samples(np_image_features)
        score = -log_density

        return {"name": self.name, "score": score}

    def evaluate_batch(
        self, image_features: List[torch.Tensor], *args, **kwargs
    ) -> list[dict[str, Any]]:

        stack = torch.cat(image_features, dim=0)

        np_stack = stack.cpu().numpy()
        neg_log_density = -self.kde.score_samples(np_stack)

        scores = neg_log_density.tolist()

        print(
            f"Calculated scores for batch with size {len(image_features)} and Shape: {image_features[0].shape}"
        )
        return [{"name": self.name, "score": score} for score in scores]

    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel

    def update(self, population: list[torch.Tensor]) -> None:

        X = torch.cat(population, dim=0).cpu().numpy()

        kde = KernelDensity(bandwidth=1.0, kernel=self.kernel, metric=self.metric)
        kde.fit(X)
        print(f"KDE Updated")
        self.kde = kde
