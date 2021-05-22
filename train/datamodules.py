import pickle as pkl
from pathlib import Path
from typing import Optional

import pytorch_lightning as pl
import yaml
from torch.utils.data import DataLoader

from train.datasets import TESSDataset


class TESSDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size: int,
        window_size: float,
    ):
        super().__init__()

        self.batch_size = batch_size
        self.windwo_size = window_size

    def setup(self, stage: Optional[str] = None):
        with open("params.yaml", 'r') as fd:
            params = yaml.safe_load(fd)

        split_file = Path(params['tess_split'])
        with open(split_file, 'rb') as f:
            split_data = pkl.load(f)

        tess_dir = Path(params['tess_mfcc'])
        self.train_set = TESSDataset(
            tess_dir, split_data['train'], self.windwo_size, rnd_window=True
        )
        self.test_set = TESSDataset(tess_dir, split_data['test'], self.windwo_size)

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_set, self.batch_size, shuffle=True, num_workers=4)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.test_set, self.batch_size, shuffle=False, num_workers=4)
