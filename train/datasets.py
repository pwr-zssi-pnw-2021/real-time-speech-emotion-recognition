import pickle as pkl
from pathlib import Path

import torch
from torch.utils.data import Dataset


class TESSDataset(Dataset):
    def __init__(
        self,
        data_dir: Path,
        file_names: set[str],
        window_size: float,
    ) -> None:
        super().__init__()

        self.window_size = window_size

        files = [f for f in data_dir.glob('**/*.pkl') if f in file_names]
        self.data = []
        for pth in files:
            with open(pth, 'rb') as f:
                d = pkl.load(f)
            self.data.append(d)

    def __getitem__(self, index) -> torch.Tensor:
        # TODO select random window of size window_size
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
