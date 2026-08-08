import random

from src.models import Noise
from src.selector_functions.base_selectorf_unction import SelectorFunction


class TournamentSelector(SelectorFunction):

    def __init__(self, tournament_size):
        assert (
            tournament_size > 1
        ), "Tournament is only possible with two or more contenders"
        self.tournament_size = tournament_size
        self.name = "TournamentSelector"

    def select(self, contenders: list[Noise]) -> Noise:
        selected = random.sample(contenders, self.tournament_size)
        winner = max(selected, key=lambda contender: contender.fitness)
        return winner
