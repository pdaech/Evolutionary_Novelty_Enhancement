from typing import List

import torch


def tensor_list_to_stack(list: List[torch.Tensor]) -> torch.Tensor:

    return torch.stack(list)
