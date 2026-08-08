import torch
from torch.nn import functional as F


class DiversityEvaluator:

    def __init__(self) -> None:
        pass

    def evalute_batch(self, embeddings):

        stack = torch.cat(embeddings, dim=0)
        n = stack.shape[0]

        if n <= 1:
            return 0.0

        t_norm = F.normalize(stack, p=2, dim=1)

        sim_matrix = torch.mm(t_norm, t_norm.t())

        dist_matrix = 1 - ((sim_matrix + 1) / 2)

        sum_dist = dist_matrix.sum()
        diag_dist = dist_matrix.diag().sum()

        avg_dist = (sum_dist - diag_dist) / (n * (n - 1))

        return avg_dist.item()
