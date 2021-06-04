from abc import ABC

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn


class SERModel(pl.LightningModule, ABC):
    def __init__(self):
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
            nn.Linear(128, 7),
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
            nn.Linear(128, 7),
        )


# class AttModel(SERModel):
#     def __init__(self, encoding_size: int = 32, dropout: float = 0.1):
#         super().__init__()

#         self.encoding_size = encoding_size

#         self.key_layer = nn.Linear(self.input_size * 20, encoding_size)
#         self.query_layer = nn.Linear(self.input_size * 20, encoding_size)
#         self.value_layer = nn.Linear(self.input_size * 20, encoding_size)

#         self.dropout = nn.Dropout(dropout)

#         self.classifier = nn.Linear(32, 7)

#     def forward(self, x):
#         keys = self.key_layer(x)
#         queries = self.query_layer(x)
#         values = self.value_layer(x)

#         att = torch.matmul(keys.unsqueeze(-1), values.unsqueeze(-1).transpose(1, 2))
#         att_scaled = att / (self.encoding_size ** 0.5)

#         sm_att = F.softmax(att_scaled, dim=-1)
#         drop_att = self.dropout(sm_att)
#         out = torch.matmul(drop_att, queries.unsqueeze(-1)).squeeze(-1)

#         c = self.classifier(out)

#         return c
