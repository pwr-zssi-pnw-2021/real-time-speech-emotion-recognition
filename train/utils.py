import pickle as pkl
import random
from pathlib import Path

import numpy as np


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
