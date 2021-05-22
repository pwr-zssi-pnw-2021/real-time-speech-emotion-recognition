import pickle as pkl
from pathlib import Path
from random import randint

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

        files = [f for f in data_dir.glob('**/*.pkl') if f.name in file_names]
        self.data = []
        for pth in files:
            with open(pth, 'rb') as f:
                d = pkl.load(f)
            self.data.append(d)

    def __getitem__(self, index) -> torch.Tensor:
        # TODO handle too big windows
        features = self.data[index]
        length = features.shape[1]
        margin = length - self.window_size

        start_idx = randint(0, margin)
        end_idx = start_idx + self.window_size

        return features[:, start_idx:end_idx]

    def __len__(self) -> int:
        return len(self.data)
