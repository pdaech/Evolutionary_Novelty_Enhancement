from typing import List

import torch

from src.mutators.base_mutator import MutationFunction


class UniformGaussianMutator(MutationFunction):

    def __init__(self, mutation_rate: float, mutation_strengh: float) -> None:
        self.mutation_rate = mutation_rate
        self.mutation_strength = mutation_strengh

    def mutate(self, embeds: torch.Tensor) -> torch.Tensor:

        device = embeds.device

        #
        num_mutations = int(torch.numel(embeds) * self.mutation_rate)
        mutation_idx = torch.randperm(torch.numel(embeds), device=device)[
            :num_mutations
        ]
        mutation_tensor = (
            torch.randn(num_mutations, device=device) * self.mutation_strength
        )

        #
        output_embeds = embeds.clone()
        flat_output_embeds = output_embeds.flatten()
        flat_output_embeds[mutation_idx] += mutation_tensor

        return flat_output_embeds.view(embeds.shape)

    def mutate_batch(self, embeds: List[torch.Tensor]) -> List[torch.Tensor]:
        return [self.mutate(embed) for embed in embeds]
