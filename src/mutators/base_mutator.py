from abc import ABC, abstractmethod
from typing import List

import torch


class MutationFunction(ABC):

    @abstractmethod
    def mutate(self, embeds: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    @abstractmethod
    def mutate_batch(self, embeds: List[torch.Tensor]) -> List[torch.Tensor]:
        raise NotImplementedError
