from typing import Optional

import numpy as np
import pytorch_lightning as pl
from torch.utils.data import DataLoader

from .datasets import SERDataset


class SERDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ):
        super().__init__()

        self.x_train = x_train
        self.y_train = y_train

        self.x_test = x_test
        self.y_test = y_test

    def setup(self, stage: Optional[str]) -> None:
        self.train_set = SERDataset(self.x_train, self.y_train)
        self.test_set = SERDataset(self.x_test, self.y_test)

        # TODO add to params ?
        self.batch_size = 256

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=4)
