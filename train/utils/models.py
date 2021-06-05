import math
from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn

from .params import get_params


class SERModel(pl.LightningModule, ABC):
    def __init__(self):
        params = get_params()
        self.cls_num = len(params['data']['emotions'])

        super().__init__()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = F.cross_entropy(y_hat, y)

        self.log('val_loss', loss, True)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=1e-4)


class LinearModel(SERModel):
    def __init__(self, data_shape: tuple[int]):
        super().__init__()

        input_size = data_shape[0]
        self.model = nn.Sequential(
            nn.Linear(input_size, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, self.cls_num),
        )


class ConvModel(SERModel):
    def __init__(self, data_shape: tuple[int, int]):
        super().__init__()

        h, w = data_shape
        self.model = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(8, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Flatten(),
            nn.Linear((h // 4) * (w // 4) * 16, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(128, self.cls_num),
        )


class PositionalEncoding(nn.Module):
    # From pytorch tutorial
    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)


class AttModel(SERModel):
    def __init__(self, data_shape: tuple[int, int]):
        super().__init__()

        self.encoding_size, self.input_size = data_shape

        self.pos_enc = PositionalEncoding(self.encoding_size, max_len=self.input_size)
        self.att_enc = nn.TransformerEncoderLayer(self.encoding_size, 2)
        self.l1 = nn.Linear(self.encoding_size * self.input_size, 32)
        self.l2 = nn.Linear(32, self.cls_num)

    def forward(self, x):
        pos_x = x + self.pos_enc(x)
        enc = self.att_enc(pos_x)
        f_enc = torch.flatten(enc, start_dim=1)
        c1 = self.l1(f_enc)
        c2 = self.l2(c1)

        return c2
