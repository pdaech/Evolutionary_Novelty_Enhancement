import random

from src.models import Noise
from src.selector_functions.base_selectorf_unction import SelectorFunction


class RouletteWheelSelector(SelectorFunction):

    def select(self, contenders: list[Noise]) -> Noise:
        contenders = [
            contender for contender in contenders if contender.fitness is not None
        ]
        total_fitness = sum(contender.fitness for contender in contenders)
        pick = random.uniform(0, total_fitness)
        current = 0
        for contender in contenders:
            current += contender.fitness
            if current > pick:
                return contender
        return random.choice(contenders)
