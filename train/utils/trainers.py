import logging
from abc import ABC, abstractmethod
from typing import Type

import numpy as np
import pytorch_lightning as pl
import torch
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from .datamodules import SERDatamodule
from .models import ConvModel, LinearModel
from .params import get_params
from .rnd_state import RND_STATE

SVM = 'svm'
TREE = 'tree'
MLP = 'mlp'
CONV = 'conv'

MODEL_MAP = {
    SVM: lambda: SVC(random_state=RND_STATE),
    TREE: lambda: DecisionTreeClassifier(random_state=RND_STATE),
    MLP: LinearModel,
    CONV: ConvModel,
}


def disable_lightning_logging() -> None:
    logging.getLogger("pytorch_lightning").setLevel(logging.ERROR)


def seed_lightning() -> None:
    params = get_params()
    pl.seed_everything(params['train']['seed'])


class Trainer(ABC):
    def __init__(
        self,
        model: str,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
    ) -> None:
        self.model_name = model

        self.x_train = x_train
        self.y_train = y_train
        self.x_test = x_test
        self.y_test = y_test

        self.reshape()
        self.setup()

    def reshape(self) -> None:
        if self.model_name != CONV:
            train_len = len(self.y_train)
            self.x_train = self.x_train.reshape((train_len, -1))

            test_len = len(self.y_test)
            self.x_test = self.x_test.reshape((test_len, -1))
        else:
            self.x_train = self.x_train[:, np.newaxis, :, :]
            self.x_test = self.x_test[:, np.newaxis, :, :]

    @abstractmethod
    def setup(self) -> None:
        pass

    @abstractmethod
    def train(self) -> None:
        pass

    @abstractmethod
    def eval(self) -> None:
        pass

    def get_metrics(self) -> list[np.ndarray]:
        # TODO metrics calculation
        # self.y_test self.y_hat
        pass


class SklearnTrainer(Trainer):
    def setup(self) -> None:
        self.model = MODEL_MAP[self.model_name]()

    def train(self) -> None:
        self.model.fit(self.x_train, self.y_train)

    def eval(self) -> None:
        self.y_hat = self.model.predict(self.x_test)


class TorchTrainer(Trainer):
    def setup(self) -> None:
        disable_lightning_logging()
        seed_lightning()

        _, _, *data_shape = self.x_train.shape
        self.model = MODEL_MAP[self.model_name](data_shape)

        self.trainer = pl.Trainer(
            gpus=1,
            max_epochs=10,
            progress_bar_refresh_rate=0,
        )

        self.datamodule = SERDatamodule(
            self.x_train,
            self.y_train,
            self.x_test,
            self.y_test,
        )

    def train(self) -> None:
        self.trainer.fit(
            self.model,
            self.datamodule,
        )

    def eval(self) -> None:
        with torch.no_grad():
            test_data = torch.tensor(self.x_test)

            results = self.model(test_data).numpy()
            self.y_hat = np.argmax(results, axis=1)


TRAINER_LOOKUP: dict[str, Type[Trainer]] = {
    SVM: SklearnTrainer,
    TREE: SklearnTrainer,
    MLP: TorchTrainer,
    CONV: TorchTrainer,
}
