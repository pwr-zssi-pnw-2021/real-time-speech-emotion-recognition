import numpy as np
from torch.utils.data import Dataset


class SERDataset(Dataset):
    def __init__(self, x: np.ndarray, y: np.ndarray) -> None:
        super().__init__()

        self.x = x
        self.y = y

    def __getitem__(self, index) -> tuple[np.ndarray, np.ndarray]:
        return self.x[index], self.y[index]

    def __len__(self) -> int:
        return len(self.y)
