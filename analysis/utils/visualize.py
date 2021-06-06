from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay


def save_conf_mat(
    mat: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    cmd = ConfusionMatrixDisplay(mat, display_labels=labels)
    _, ax = plt.subplots(figsize=(10, 10))
    plt.title(title, fontsize=20)
    cmd.plot(ax=ax)
    plt.savefig(path)


def save_diff_mat(
    mat: np.ndarray,
    labels: list[str],
    title: str,
    path: Path,
) -> None:
    plt.pcolormesh(mat, edgecolors='black', linewidth=1)

    ax = plt.gca()
    ax.set_aspect('equal')

    t = np.arange(len(labels)) + 0.5
    plt.yticks(t, labels)
    plt.xticks(t, labels, rotation=-30, ha='left', rotation_mode='anchor')

    ax.invert_yaxis()
    plt.gcf().set_size_inches(8, 8)

    plt.title(title, fontsize=20)
    plt.savefig(path)
