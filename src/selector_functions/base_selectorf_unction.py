from abc import ABC, abstractmethod
from typing import List, Dict, Any

import torch

from src.models import Noise


class SelectorFunction(ABC):

    @abstractmethod
    def select(self, contenders: list[Noise]) -> Noise:
        raise NotImplementedError
