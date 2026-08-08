import os
from pathlib import Path
from typing import Any, List
from torchkde import KernelDensity
import torch
import numpy as np
from src.evaluators.base_evaluator import Evaluator
from src.huggingface_models.image_embedding.blip2_embedding import Blip2EmbeddingModel
from src.utils.pca import PCA


class KernelDensityEstimationEvaluator(Evaluator):
    def __init__(
        self,
        prompt: str,
        rule_of_thumb: str = "silverman",
        bandwidth: float | None = None,
        kernel: str = "gaussian",
        metric_path: Path | None = None,
        K: int = 128,
    ) -> None:
        self.kernel: str = kernel
        self.prompt: str = prompt
        self.K: int = K
        if metric_path is None:
            base_path = Path(os.environ["BASE_PATH"])
            metrics_path = Path(os.environ["BLIP_2_BASELINE"])
            self.metric_path = base_path / metrics_path / self.prompt.replace(" ", "_")
        else:
            self.metric_path = metric_path
        print(self.metric_path)

        glob = self.metric_path.glob("*.pt")
        data = []
        for p in glob:
            dp = torch.load(
                str(p),
                map_location=torch.device("cuda"),
                weights_only=False,
            )
            data.append(dp)

        if len(data) == 0:
            raise FileNotFoundError(f"No metric found at {self.metric_path}")

        data_stack = torch.cat(data, dim=0)
        data_stack = data_stack.to(torch.float32)
        self.health_check(data_stack)
        self.pca = PCA(data_stack)

        self.reduced_data_stack = self.pca.reduce_embeddings(data_stack, self.K)
        std_check = torch.std(self.reduced_data_stack, dim=0).min().item()

        if bandwidth is None:

            N = self.reduced_data_stack.size(0)

            bandwidth = self.calculate_bandwidth(
                self.reduced_data_stack, N, rule_of_thumb
            )

        self.kde = KernelDensity(kernel=kernel, bandwidth=bandwidth)
        self.kde.fit(self.reduced_data_stack)
        self.name = "KernelDensityEstimation"

    def health_check(self, embeddings: torch.Tensor):

        std_per_dim = torch.std(embeddings, dim=0)
        min_std = std_per_dim.min().item()
        max_std = std_per_dim.max().item()
        print(f"DEBUG: Min STD über 1408 Dimensionen: {min_std:.8f}")
        print(f"DEBUG: Max STD über 1408 Dimensionen: {max_std:.8f}")
        first_sample = embeddings[0:1]  # Size [1, 1408]
        diff = embeddings - first_sample
        distances_sq = torch.sum(diff**2, dim=1)
        max_non_zero_distance = distances_sq.max().item()
        print(
            f"DEBUG: Maximale Distanz² zum ersten Sample: {max_non_zero_distance:.8f}"
        )

    def calculate_bandwidth(
        self, embeddings: torch.Tensor, N: int, rule_of_thumb: str = "silverman"
    ):
        embeddings = embeddings.cpu()
        std_dims = torch.std(embeddings, dim=0)
        sigma_median = np.median(std_dims.numpy())

        N_exponent = N ** (-1 / 5)

        if rule_of_thumb == "silverman":
            h = 0.9 * sigma_median * N_exponent
        else:
            h = 1.06 * sigma_median * N_exponent

        return h

    def evaluate(self, embeds: torch.Tensor, *args, **kwargs) -> dict[str, float]:

        embeds = embeds.to(self.reduced_data_stack.dtype).to(
            self.reduced_data_stack.device
        )
        reduced__embeds = self.pca.reduce_embeddings(embeds, self.K)
        log_prob = self.kde.score_samples(reduced__embeds).item()
        return {"name": self.name, "score": log_prob}

    def evaluate_batch(
        self, embeds: List[torch.Tensor], *args, **kwargs
    ) -> list[dict[str, float]]:

        print(self.reduced_data_stack.size())
        stack = torch.stack(embeds)
        stack = stack.to(self.reduced_data_stack.dtype).to(
            self.reduced_data_stack.device
        )
        self.health_check(stack)
        print(embeds[0].size())
        print(stack.size())
        reduced__stack = self.pca.reduce_embeddings(stack, self.K)
        log_prob = self.kde.score_samples(reduced__stack)
        novelty_scores: List[float] = log_prob.tolist()

        return [{"name": self.name, "score": log_prob} for log_prob in novelty_scores]

    @classmethod
    def need(cls) -> type:
        return Blip2EmbeddingModel
