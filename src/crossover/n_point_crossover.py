from dataclasses import dataclass

import torch

from src.crossover.base_crossover import CrossoverFunction


class NPointRandomCrossoverFunction(CrossoverFunction):
    def __init__(self, n_points: int = 3) -> None:
        super().__init__()
        self.n_points = n_points
        self.name = f"{n_points}_PointRandomCrossoverFunction"

    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Perform uniform crossover between parent_1 and parent_2 tensors.

        Args:
            parent_1 (torch.Tensor): [B,C,H,W] First parent tensor
            parent_2 (torch.Tensor): [B,C,H,W] Second parent tensor
            *args: Additional positional arguments (not used)
            **kwargs (float): Additional keyword arguments with optional 'n_point' parameter

        Returns:
            torch.Tensor: [B,C,H,W] Child tensor created with n+1 random sized segment mix of parents
        """
        assert (
            parent_1.shape == parent_2.shape
        ), "Parents must have the same size for crossover."

        n_point: int = kwargs.pop("n_point", self.n_points)
        _, C, H, W = parent_1.shape
        length = H * W
        parent_1_squeeze = parent_1.squeeze(0)
        parent_2_squeeze = parent_2.squeeze(0)

        parent_1_flat = parent_1_squeeze.reshape(C, length)
        parent_2_flat = parent_2_squeeze.reshape(C, length)
        child_flat = torch.empty_like(parent_1_flat)

        points = torch.sort(torch.randint(low=1, high=length, size=(n_point,)))[0]

        start = 0
        take_from_parent_1 = True
        for point in points:
            if take_from_parent_1:
                child_flat[:, start:point] = parent_1_flat[:, start:point]
            else:
                child_flat[:, start:point] = parent_2_flat[:, start:point]
            start = point
            take_from_parent_1 = not take_from_parent_1

        if take_from_parent_1:
            child_flat[:, start:] = parent_1_flat[:, start:]
        else:
            child_flat[:, start:] = parent_2_flat[:, start:]

        child_squeeze = child_flat.reshape(C, H, W)
        child = child_squeeze.unsqueeze(0)
        return child


class NPointEqualCrossoverFunction(CrossoverFunction):

    def __init__(self, n_points: int = 3) -> None:
        super().__init__()
        self.n_points = n_points
        self.name = f"{n_points}_PointEqualCrossoverFunction"

    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Perform uniform crossover between parent_1 and parent_2 tensors.

        Args:
            parent_1 (torch.Tensor): [B,C,H,W] First parent tensor
            parent_2 (torch.Tensor): [B,C,H,W] Second parent tensor
            *args: Additional positional arguments (not used)
            **kwargs (float): Additional keyword arguments with optional 'n_point' parameter

        Returns:
            torch.Tensor: [B,C,H,W] Child tensor created with n+1 random sized segment mix of parents
        """
        assert (
            parent_1.shape == parent_2.shape
        ), "Parents must have the same size for crossover."

        n_point: int = kwargs.pop("n_point", self.n_points)
        _, C, H, W = parent_1.shape
        length = H * W
        parent_1_squeeze = parent_1.squeeze(0)
        parent_2_squeeze = parent_2.squeeze(0)

        parent_1_flat = parent_1_squeeze.reshape(C, length)
        parent_2_flat = parent_2_squeeze.reshape(C, length)
        child_flat = torch.empty_like(parent_1_flat)

        num_segments = n_point + 1
        segment_length = length // num_segments

        points = torch.arange(1, num_segments, device=parent_1.device) * segment_length

        start = 0
        take_from_parent_1 = True
        for point in points:
            if take_from_parent_1:
                child_flat[:, start:point] = parent_1_flat[:, start:point]
            else:
                child_flat[:, start:point] = parent_2_flat[:, start:point]
            start = point
            take_from_parent_1 = not take_from_parent_1

        if take_from_parent_1:
            child_flat[:, start:] = parent_1_flat[:, start:]
        else:
            child_flat[:, start:] = parent_2_flat[:, start:]

        child_squeeze = child_flat.reshape(C, H, W)
        child = child_squeeze.unsqueeze(0)
        return child


class NPointRandomPixelCrossoverFunction(CrossoverFunction):

    def __init__(self, n_points: int = 3) -> None:
        super().__init__()
        self.n_points = n_points
        self.name = f"{n_points}_PointRandomPixelCrossoverFunction"

    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Perform n-point crossover between parent_1 and parent_2 tensors using broadcasting.

        Args:
            parent_1 (torch.Tensor): [B,C,H,W] First parent tensor
            parent_2 (torch.Tensor): [B,C,H,W] Second parent tensor
            **kwargs: Optional 'n_point' parameter

        Returns:
            torch.Tensor: [B,C,H,W] Child tensor
        """

        assert parent_1.shape == parent_2.shape, "Parents must have the same size."

        is_batched = parent_1.ndim == 4
        if not is_batched:
            parent_1 = parent_1.unsqueeze(0)
            parent_2 = parent_2.unsqueeze(0)

        B, C, H, W = parent_1.shape
        L = H * W
        device = parent_1.device

        n_point = kwargs.pop("n_point", self.n_points)

        crossover_points = (
            torch.randint(1, L, (B, n_point), device=device).sort(dim=1).values
        )
        crossover_points = crossover_points.unsqueeze(-1)  # [B, n_point, 1]

        indices = torch.arange(L, device=device).view(1, 1, L)

        passed_points_count = (indices >= crossover_points).sum(dim=1)  # [B, L]

        mask_flat = (passed_points_count % 2) == 0  # [B, L]

        mask = mask_flat.view(B, 1, H, W)

        child = torch.where(mask, parent_1, parent_2)

        if not is_batched:
            child = child.squeeze(0)

        return child


class NPointEqualPixelCrossoverFunction(CrossoverFunction):

    def __init__(self, n_points: int = 3) -> None:
        super().__init__()
        self.n_points = n_points
        self.name = f"{n_points}_PointEqualPixelCrossoverFunction"

    def crossover(
        self, parent_1: torch.Tensor, parent_2: torch.Tensor, *args, **kwargs
    ) -> torch.Tensor:
        """
        Perform n-point crossover between parent_1 and parent_2 tensors using broadcasting.

        Args:
            parent_1 (torch.Tensor): [B,C,H,W] First parent tensor
            parent_2 (torch.Tensor): [B,C,H,W] Second parent tensor
            **kwargs: Optional 'n_point' parameter

        Returns:
            torch.Tensor: [B,C,H,W] Child tensor
        """

        assert parent_1.shape == parent_2.shape, "Parents must have the same size."

        is_batched = parent_1.ndim == 4
        if not is_batched:
            parent_1 = parent_1.unsqueeze(0)
            parent_2 = parent_2.unsqueeze(0)

        B, C, H, W = parent_1.shape
        L = H * W
        device = parent_1.device

        n_point = kwargs.pop("n_point", self.n_points)

        num_segments = n_point + 1
        segment_length = L // num_segments

        points = torch.arange(1, num_segments, device=device) * segment_length
        crossover_points = points.view(1, n_point, 1)

        indices = torch.arange(L, device=device).view(1, 1, L)

        passed_points_count = (indices >= crossover_points).sum(dim=1)  # [B, L]

        mask_flat = (passed_points_count % 2) == 0  # [B, L]

        mask = mask_flat.view(B, 1, H, W)

        child = torch.where(mask, parent_1, parent_2)

        if not is_batched:
            child = child.squeeze(0)

        return child
