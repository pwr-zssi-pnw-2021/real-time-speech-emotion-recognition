import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import yaml
from torch import nn
from torchmetrics.classification import Accuracy


class TESSLinearModel(pl.LightningModule):
    def __init__(self):
        super().__init__()

        with open("params.yaml", 'r') as fd:
            params = yaml.safe_load(fd)
        input_size = params['train']['window_size']

        self.model = nn.Sequential(
            nn.Linear(input_size * 20, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 7),
            nn.Softmax(),
        )
        self.train_acc = Accuracy()
        self.val_acc = Accuracy()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.train_acc(y_hat, y)
        self.log('train_acc', self.train_acc, on_step=True, on_epoch=False, prog_bar=True)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.val_acc(y_hat, y)
        self.log('val_acc', self.val_acc, on_step=True, on_epoch=True, prog_bar=True)

        self.log('val_loss', loss, True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)
