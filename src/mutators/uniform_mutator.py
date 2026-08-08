from typing import List
import torch

from src.mutators.base_mutator import MutationFunction


class UniformMutator(MutationFunction):

    def __init__(self, mutation_rate: float, std: float = 1.0) -> None:
        """
        Args:
            mutation_rate: Probability (0.0 to 1.0) that a single value will be mutated (replaced).
            std: Standard deviation for the newly sampled Gaussian noise.
        """
        self.mutation_rate = mutation_rate
        self.std = std

    def mutate(self, embeds: torch.Tensor) -> torch.Tensor:

        new_noise = torch.randn_like(embeds) * self.std

        mask = torch.rand_like(embeds) < self.mutation_rate

        output_embeds = torch.where(mask, new_noise, embeds)

        return output_embeds

    def mutate_batch(self, embeds: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.mutate(embed) for embed in embeds]
