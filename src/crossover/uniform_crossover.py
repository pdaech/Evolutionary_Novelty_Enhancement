from dataclasses import dataclass

import torch

from src.crossover.base_crossover import CrossoverFunction


@dataclass
class UniformCrossover(CrossoverFunction):

    swap_rate: float = 0.5
    name = "UniformCrossover"

    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Perform uniform crossover between parent_1 and parent_2 tensors.

        Args:
            parent_1 (torch.Tensor): [C,H,W] First parent tensor
            parent_2 (torch.Tensor): [C,H,W] Second parent tensor
            *args: Additional positional arguments (not used)
            **kwargs (float): Additional keyword arguments with optional 'swap_rate' parameter

        Returns:
            torch.Tensor: [C,H,W] Child tensor created with weighted mix of parents
        """
        assert (
            parent_1.shape == parent_2.shape
        ), "Noises must have the same size for crossover."

        swap_rate = kwargs.pop("swap_rate", self.swap_rate)
        parent_1 = parent_1.to("cuda")
        parent_2 = parent_2.to("cuda")
        crossover_mask = torch.rand(parent_1.shape) < swap_rate
        crossover_mask = crossover_mask.to("cuda")
        child = torch.where(crossover_mask, parent_1, parent_2)

        return child


@dataclass
class UniformPixelCrossover(CrossoverFunction):

    swap_rate: float = 0.5
    name = "UniformPixelCrossover"

    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Perform uniform crossover between parent_1 and parent_2 tensors at pixel level.
        Preserves channel consistency (all channels switch together for a given pixel).

        Args:
            parent_1 (torch.Tensor): [..., C, H, W] First parent tensor (supports 3D or 4D with Batch)
            parent_2 (torch.Tensor): [..., C, H, W] Second parent tensor
            *args: Additional positional arguments
            **kwargs: Additional keyword arguments

        Returns:
            torch.Tensor: Child tensor with same shape as parents
        """

        assert (
            parent_1.shape == parent_2.shape
        ), "Parents must have the same size for crossover."

        device = parent_1.device
        swap_rate = kwargs.pop("swap_rate", self.swap_rate)

        mask_shape = list(parent_1.shape)
        mask_shape[-3] = 1

        crossover_mask = torch.rand(mask_shape, device=device) < swap_rate

        child = torch.where(crossover_mask, parent_1, parent_2)

        return child
