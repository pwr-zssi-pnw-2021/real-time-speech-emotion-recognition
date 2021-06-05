import numpy as np
import torch
from torch.utils.data import Dataset


class SERDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()

        self.x = torch.tensor(x).float()
        self.y = torch.tensor(y).long()

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.y)
