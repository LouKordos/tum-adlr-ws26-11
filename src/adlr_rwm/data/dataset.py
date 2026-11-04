from pathlib import Path

from torch import Tensor
from torch.utils.data import Dataset


class RWMDataset(Dataset):
    def __init__(
        self,
        data_dir: str | Path,
    ):
        self.data_dir = Path(data_dir)

        # Load recordings / build sample index
        self.samples = []

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, Tensor]: ...
