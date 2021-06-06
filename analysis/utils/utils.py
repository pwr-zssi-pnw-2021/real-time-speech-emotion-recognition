import pickle as pkl
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm

from .tests import ttest
from .visualize import save_conf_mat, save_diff_mat

MODEL = 'model'
FEATURES = 'features'
ACCURACY = 'accuracy'
PRECISION = 'precision'
RECALL = 'recall'
F1 = 'f1'
CONFUSION_MATRIX = 'confusion matrix'


def get_params() -> dict:
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    return params


def get_emotions() -> list[str]:
    params = get_params()

    return params['data']['emotions']


def load_results() -> pd.DataFrame:
    params = get_params()

    results_dir = Path(params['train']['results_dir'])
    r_files = results_dir.glob('*.pkl')

    results = {
        MODEL: [],
        FEATURES: [],
        ACCURACY: [],
        PRECISION: [],
        RECALL: [],
        F1: [],
        CONFUSION_MATRIX: [],
    }
    for file in r_files:
        with open(file, 'rb') as f:
            data = pkl.load(f)

        model, feature = file.stem.split('_')

        results[MODEL].append(model)
        results[FEATURES].append(feature)
        results[ACCURACY].append(data['acc'])
        results[PRECISION].append(data['prec'])
        results[RECALL].append(data['rec'])
        results[F1].append(data['f1'])
        results[CONFUSION_MATRIX].append(data['conf'])

    return pd.DataFrame(results).sort_values([MODEL, FEATURES])


def combine(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    combine_cols = [ACCURACY, PRECISION, RECALL, F1, CONFUSION_MATRIX]
    for _, row in data.iterrows():
        for col in combine_cols:
            row[col] = np.stack(row[col])

    return data


def mean_over_folds(data: pd.DataFrame) -> pd.DataFrame:
    data = data.copy()
    mean_cols = [ACCURACY, PRECISION, RECALL, F1, CONFUSION_MATRIX]
    for _, row in data.iterrows():
        for col in mean_cols:
            row[col] = np.mean(row[col], axis=0).round(2)

    return data


def get_labels(data: pd.DataFrame) -> list[str]:
    labels = []
    for _, r in data.iterrows():
        l = f'{r[MODEL]} ({r[FEATURES]})'
        labels.append(l)

    return labels


def get_acc(data: pd.DataFrame) -> np.ndarray:
    return np.stack([r[ACCURACY] for _, r in data.iterrows()])


def get_mean_metric(data: pd.DataFrame, metric: str) -> np.ndarray:
    return np.stack([r[metric].mean(axis=0) for _, r in data.iterrows()])


def plot_metric(
    metric: str,
    data: pd.DataFrame,
    labels: list[str],
    out_dir: Path,
) -> None:
    if metric == ACCURACY:
        d = get_acc(data)
    else:
        d = get_mean_metric(data, metric)

    plot_file = out_dir / f'{metric}_diff.png'
    diff = ttest(d)

    save_diff_mat(
        diff,
        labels,
        f'Model comparison base on {metric}',
        plot_file,
    )


def analyze() -> None:
    params = get_params()
    data = load_results()

    plots_dir = Path(params['analysis']['plots'])
    plots_dir.mkdir(exist_ok=True)

    c_data = combine(data)
    m_data = mean_over_folds(c_data)

    labels = get_labels(c_data)

    print('Plotting metrics:')
    for m in tqdm([ACCURACY, PRECISION, RECALL, F1]):
        plot_metric(m, c_data, labels, plots_dir)

    print('Plotting confision matrices:')
    emotions = get_emotions()
    for _, r in tqdm(m_data.iterrows(), total=len(m_data)):
        model = r[MODEL]
        features = r[FEATURES]
        file_path = plots_dir / f'{model}_{features}_confusion.png'

        save_conf_mat(
            r[CONFUSION_MATRIX],
            emotions,
            f'Confusion matrix\n{model} model nad {features} features',
            file_path,
        )
