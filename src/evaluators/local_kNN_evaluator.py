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

import numpy as np
from sklearn.neighbors import NearestNeighbors


class LocalkNNEvaluator(GlobalMaxMeanDivergenceEvaluator):

    def __init__(
        self, prompt, device: str, n_neighbors: int = 10, metric: str = "cosine"
    ):
        self.device = device
        self.name = "LocalkNN"
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn = None

    def evaluate(self, image_features: torch.Tensor, *args, **kwargs) -> dict[str, Any]:

        np_features = image_features.cpu().numpy()
        distances, _ = self.knn.kneighbors(np_features)

        radius = distances[:, self.n_neighbors - 1]

        density = self.n_neighbors / radius
        epsilon = 1e-8
        score = -np.log(density + epsilon)
        return {"name": self.name, "score": score}

    def evaluate_batch(
        self, image_features: List[torch.Tensor], *args, **kwargs
    ) -> list[dict[str, Any]]:

        stack = torch.cat(image_features, dim=0).cpu().numpy()
        distances, _ = self.knn.kneighbors(stack, n_neighbors=self.n_neighbors + 1)
        actual_distances = distances[:, 1:]
        avg_distances = np.mean(actual_distances, axis=1)
        print(f"Calculated kNN density scores for batch of size {len(image_features)}")
        return [{"name": self.name, "score": score} for score in avg_distances]

    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel

    def update(self, population: list[torch.Tensor]) -> None:

        X = torch.cat(population, dim=0).cpu().numpy()

        actual_neighbors = min(self.n_neighbors, len(X) - 1)

        self.knn = NearestNeighbors(n_neighbors=actual_neighbors, metric=self.metric)
        self.knn.fit(X)
        print(
            f"kNN model updated with n_neighbors={self.n_neighbors} and metric={self.metric}"
        )


class LocalArchivkNNEvaluator(GlobalMaxMeanDivergenceEvaluator):

    def __init__(
        self,
        prompt,
        device: str,
        archive_size: int = 200,
        archive_threshold: float = 0.1,
        n_neighbors: int = 10,
        metric: str = "cosine",
    ):
        self.device = device
        self.name = "LocalArchivkNNEvaluator"
        self.n_neighbors = n_neighbors
        self.metric = metric
        self.knn = None

        self.archive_size = archive_size
        self.archive_threshold = archive_threshold
        self.archive = deque(maxlen=archive_size)

    def evaluate(self, image_features: torch.Tensor, *args, **kwargs) -> dict[str, Any]:

        np_features = image_features.cpu().numpy()
        if np_features.ndim == 1:
            np_features = np_features.reshape(1, -1)

        distances, _ = self.knn.kneighbors(
            np_features, n_neighbors=self.n_neighbors + 1
        )

        score = np.mean(distances[:, 1:])
        return {"name": self.name, "score": float(score)}

    def evaluate_batch(
        self, image_features: List[torch.Tensor], *args, **kwargs
    ) -> list[dict[str, Any]]:

        stack = torch.cat(image_features, dim=0).cpu().numpy()
        distances, _ = self.knn.kneighbors(stack, n_neighbors=self.n_neighbors + 1)
        actual_distances = distances[:, 1:]
        avg_distances = np.mean(actual_distances, axis=1)

        high_score_indices = np.where(avg_distances > self.archive_threshold)[0]

        for idx in high_score_indices:
            feature = torch.from_numpy(stack[idx]).unsqueeze(0).detach().clone()
            self.archive.append(feature)

        print(f"Calculated kNN density scores for batch of size {len(image_features)}")
        return [{"name": self.name, "score": score} for score in avg_distances]

    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel

    def update(self, population: list[torch.Tensor]) -> None:

        current_pop_stack = torch.cat(population, dim=0)

        if len(self.archive) > 0:

            archive_stack = torch.cat(list(self.archive), dim=0).to(self.device)

            X = torch.cat([current_pop_stack, archive_stack], dim=0)
            print(
                f"Updating mean with Population ({len(population)}) + Archive ({len(self.archive)})"
            )
        else:
            X = current_pop_stack.cpu().numpy()
            print(
                f"Updating mean with Population ({len(population)}) only (Archive empty)"
            )

        actual_neighbors = min(self.n_neighbors, len(X) - 1)

        self.knn = NearestNeighbors(n_neighbors=actual_neighbors, metric=self.metric)
        self.knn.fit(X)
        print(
            f"kNN model updated with n_neighbors={self.n_neighbors} and metric={self.metric}"
        )
