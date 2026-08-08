import random

from src.models import Noise
from src.selector_functions.base_selectorf_unction import SelectorFunction


class RankSelector(SelectorFunction):

    def __init__(self, selection_pressure: float) -> None:
        assert selection_pressure > 1.0, "The Bias should not be less than 1.0"
        self.selection_pressure = selection_pressure
        self.name = "RankSelector"

    def select(self, contenders: list[Noise]) -> Noise:

        contenders.sort(key=lambda contender: contender.fitness, reverse=True)
        total = len(contenders)

        if total == 1:
            return contenders[0]

        rank_weights = [
            (self.selection_pressure - 2.0)
            * (self.selection_pressure - 1.0)
            * (i / total - 1)
            for i in range(total)
        ]
        chosen = random.choices(range(total), weights=rank_weights, k=1)[0]
        winner = contenders[chosen]

        return winner
