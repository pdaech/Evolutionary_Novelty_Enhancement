import torch


class PCA:

    def __init__(self, X: torch.Tensor) -> None:
        X = X.to(torch.float32)
        self.mean = X.mean(dim=0, keepdim=True)
        X_centered = X - self.mean
        _, _, self.Vh = torch.linalg.svd(X_centered, full_matrices=False)

    def reduce_embeddings(self, X: torch.Tensor, K: int = 128) -> torch.Tensor:

        X = X.to(torch.float32)
        assert (
            K <= self.Vh.shape[0]
        ), f"K={K} is bigger than the original dimension {self.Vh.shape[0]}"
        principal_components = self.Vh[:K, :]
        X_centered = X - self.mean
        X_reduced = torch.matmul(X_centered, principal_components.T)

        return X_reduced
