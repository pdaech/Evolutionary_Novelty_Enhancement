from abc import ABC, abstractmethod
from typing import List

import torch


class CrossoverFunction(ABC):

    @abstractmethod
    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        raise NotImplementedError
