from abc import ABC, abstractmethod
from typing import Any

import torch


class Evaluator(ABC):
    @abstractmethod
    def evaluate(self, image_features: torch.Tensor, *args, **kwargs) -> float:
        raise NotImplementedError

    @abstractmethod
    def evaluate_batch(
        self, image_features: list[torch.Tensor], *args, **kwargs
    ) -> list[float]:
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def need(cls) -> type | list[type] | None:
        raise NotImplementedError

    def update(self, population: list[torch.Tensor]) -> None:
        pass

    def config_metadata(self) -> dict[str, Any]:
        """Return JSON-serializable evaluator details for experiment provenance."""
        return {}
