from pathlib import Path

import torch


class Loader:
    pass


class TensorLoader(Loader):

    def __init__(self, base_path: Path):

        self.base_path = base_path

    def load_file(self, path: Path, weights_only=True):
        query = self.base_path / path.name
        return torch.load(path, weights_only=weights_only)

    def load_files(
        self,
        path: Path,
        query_filter: str = "*.pt",
        device: str = "cuda",
        weights_only=True,
    ) -> list[torch.Tensor]:

        output = []
        query = self.base_path / path.name
        glob = path.glob(query_filter)
        for p in glob:
            print(p)
            t = torch.load(
                str(p),
                map_location=torch.device(device),
                weights_only=weights_only,
            )
            output.append(t)

        return output
