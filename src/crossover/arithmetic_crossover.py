from dataclasses import dataclass

import torch
import math
from .base_crossover import CrossoverFunction


@dataclass
class ArithmeticCrossoverFunction(CrossoverFunction):
    alpha: float = 0.5
    name = "ArithmeticCrossoverFunction"

    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Perform weighted arithmetic crossover between two parent tensors.
        Args:
            parent_1 (torch.Tensor): [C,H,W] First parent tensor
            parent_2 (torch.Tensor): [C,H,W] Second parent tensor
            *args: Additional positional arguments (not used)
            **kwargs: Additional keyword arguments with optional 'alpha' parameter

        Returns:
            torch.Tensor: [C,H,W] Child tensor created with weighted sum of parents

        """
        alpha = kwargs.pop("alpha", self.alpha)
        parent_2 = parent_2.to(parent_1.device)
        return alpha * parent_1 + (1 - alpha) * parent_2


@dataclass
class ArithmeticRenormedCrossoverFunction(CrossoverFunction):
    alpha: float = 0.5
    name = "ArithmeticCrossoverFunction"

    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Perform variance-preserved weighted arithmetic crossover.
        """
        alpha = kwargs.pop("alpha", self.alpha)
        parent_2 = parent_2.to(parent_1.device)

        child = alpha * parent_1 + (1 - alpha) * parent_2

        renorm_factor = math.sqrt(alpha**2 + (1 - alpha) ** 2)

        return child / renorm_factor
