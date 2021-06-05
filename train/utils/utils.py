import pickle as pkl
import random
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .rnd_state import RND_STATE, get_params
from .trainers import TRAINER_LOOKUP


def load_window_data(
    index_file: Path,
    window_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    with open(index_file, 'rb') as f:
        paths, c = pkl.load(f)

    all_features = []
    for path in paths:
        with open(path, 'rb') as f:
            d = pkl.load(f)
        all_features.append(d)

    f_features, f_classes = filter_short(all_features, c, window_size)
    w_features = create_windows(f_features, window_size)

    classes = np.array(f_classes)
    features = np.stack(w_features)

    return features, classes


def filter_short(
    features: list[np.ndarray],
    classes: list[int],
    threshold: int,
) -> tuple[list[np.ndarray], list[int]]:
    f_features = []
    f_classes = []
    for f, c in zip(features, classes):
        if f.shape[1] >= threshold:
            f_features.append(f)
            f_classes.append(c)

    return f_features, f_classes


def create_windows(features: list[np.ndarray], window_size: int) -> list[np.ndarray]:
    w_features = []
    for f in features:
        f_len = f.shape[1]
        margin = f_len - window_size

        start_pos = random.randint(0, margin)
        end_pos = start_pos + window_size

        w_f = f[:, start_pos:end_pos]
        w_features.append(w_f)

    return w_features


def get_index_file(features: str, params: dict) -> Path:
    index_dir = Path(params['data']['global_index_dir'])
    index_file = index_dir / f'{features}.index'

    if not index_file.exists():
        raise FileNotFoundError(f'Index file not found: {index_file}')

    return index_file


def save_metrics(
    metrics: tuple[np.ndarray],
    out_file: Path,
) -> None:

    out_file.parent.mkdir(exist_ok=True)
    with open(out_file, 'wb') as f:
        pkl.dump(metrics, f, pkl.HIGHEST_PROTOCOL)


def train_model(
    model: str,
    features: str,
    window_size: int,
    out_file_path: str,
) -> None:
    params = get_params()

    index_file = get_index_file(features, params)
    x, y = load_window_data(index_file, window_size)

    out_file = Path(out_file_path)

    trainer_cls = TRAINER_LOOKUP[model]

    folds = params['train']['folds']
    kf = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=RND_STATE,
    )
    for train_idx, test_idx in kf.split(x, y):
        x_train, y_train = x[train_idx], y[train_idx]
        x_test, y_test = x[test_idx], y[test_idx]

        trainer = trainer_cls(
            model,
            x_train,
            y_train,
            x_test,
            y_test,
        )

        trainer.train()
        trainer.eval()
        metrics = trainer.get_metrics()

        save_metrics(metrics, out_file)
