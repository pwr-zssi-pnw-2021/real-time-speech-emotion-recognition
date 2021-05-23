from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torchmetrics.classification import Accuracy


class TESSModel(pl.LightningModule, ABC):
    def __init__(self):
        super().__init__()

        with open("params.yaml", 'r') as fd:
            params = yaml.safe_load(fd)

        self.input_size = params['train']['window_size']

        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.train_acc(F.softmax(y_hat, dim=1), y)
        self.log(
            'train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.val_acc(F.softmax(y_hat, dim=1), y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

        self.log('val_loss', loss, True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class TESSLinearModel(TESSModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(self.input_size * 20, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )


class TESSConvModel(TESSModel):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear(12 * 5 * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
        )
