import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from scipy.stats import ttest_rel
from tqdm import tqdm

from .tabularize import generate_table
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


def ttest(data: np.ndarray, alpha: float) -> np.ndarray:
    s = len(data)

    t_stats = np.zeros((s, s))
    p_vals = np.zeros((s, s))

    for i in range(s):
        for j in range(s):
            t_stats[i, j], p_vals[i, j] = ttest_rel(data[i], data[j])

    res = (t_stats > 0) & (p_vals <= alpha)
    better = np.argwhere(res)

    return better


def ttest_analyze(
    data: pd.DataFrame,
    metrics: list[str],
    alpha: float,
) -> list[np.ndarray]:
    all_better = []
    for m in metrics:
        metric_m = np.stack(data[m])

        metric_better = ttest(metric_m, alpha)
        all_better.append(metric_better)

    return all_better


def mean_metrics_over_classes(data: pd.DataFrame, metrics: list[str]) -> pd.DataFrame:
    data = data.copy()
    for m in metrics:
        data[m] = data[m].apply(lambda x: np.mean(x, axis=1))

    return data


def generate_table(
    data: pd.DataFrame,
    better: list[np.ndarray],
    metrics: list[str],
    feature: str,
    out_dir: Path,
) -> None:
    all_df = []
    for b, m in zip(better, metrics):
        f_data = data[[MODEL, m]]
        f_data[m] = f_data[m].apply(np.mean)

        res_table = f_data.round(3).T.reset_index()
        res_table.columns = res_table.iloc[0]
        res_table = res_table.drop(0)

        better_strings = ['']
        for i in range(len(data)):
            mask = b[:, 0] == i
            better_than = b[mask][:, 1] + 1

            if not len(better_than):
                b_str = '--'
            else:
                b_str = ', '.join(map(str, better_than))
            better_strings.append(b_str)
        better_df = pd.DataFrame(better_strings, index=res_table.columns).T

        table = pd.concat((res_table, better_df))
        all_df.append(table)

    full_table = pd.concat(all_df)
    f_path = out_dir / f'ttest_{feature}.tex'

    full_table.to_latex(f_path, index=False)


def plot_confusion() -> None:
    params = get_params()
    data = load_results()
    f_data = data[[MODEL, FEATURES, CONFUSION_MATRIX]]
    emotions = get_emotions()

    out_dir = Path(params['analysis']['plots'])
    out_dir.mkdir(exist_ok=True)

    for _, r in f_data.iterrows():
        model = r[MODEL]
        features = r[FEATURES]

        matrix = np.stack(r[CONFUSION_MATRIX]).mean(axis=0)

        fig, ax = plt.subplots()
        im = ax.imshow(matrix)

        color_th = (matrix.max() - matrix.min()) / 2

        ax.figure.colorbar(im, ax=ax)

        ticks = np.arange(len(emotions))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)

        ax.set_xticklabels(emotions)
        ax.set_yticklabels(emotions)

        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')

        plt.setp(ax.get_xticklabels(), rotation=30, ha='right', rotation_mode='anchor')

        for i in ticks:
            for j in ticks:
                if matrix[i, j] > color_th:
                    ax.text(j, i, matrix[i, j], ha='center', va='center', color='k')
                else:
                    ax.text(
                        j, i, matrix[i, j], ha='center', va='center', color='yellow'
                    )

        fig.set_size_inches(10, 8)
        fig.tight_layout()

        f_path = out_dir / f'confusion_{model}_{features}.png'
        plt.savefig(f_path)


def plot_radar() -> None:
    data = load_results()
    params = get_params()
    metrics = [ACCURACY, PRECISION, RECALL, F1]
    features = params['data']['features']
    out_dir = Path(params['analysis']['plots'])
    out_dir.mkdir(exist_ok=True)

    data = mean_metrics_over_classes(data, [PRECISION, RECALL, F1])
    for m in metrics:
        data[m] = data[m].apply(np.mean)

    for f in features:
        f_data = data[data[FEATURES] == f]

        plots = []
        positions = [(0, 0), (0, 3), (3, 0), (3, 3)]
        for (_, r), p in zip(f_data.iterrows(), positions):
            acc = r[ACCURACY]
            pre = r[PRECISION]
            rec = r[RECALL]
            f1 = r[F1]

            plot = f'\t\t\\startcord{{{acc}}}{{{pre}}}{{{rec}}}{{{f1}}}{{{p[0]}}}{{{p[1]}}}{{yellow}}'
            plots.append(plot)

        j_plots = '\n'.join(plots)

        figure = f"""
        \\begin{{figure}}
        \t\\centering
        \t\\begin{{tikzpicture}}[scale=2.0]
        {j_plots}
        \t\\end{{tikzpicture}}
        \t\\caption{{Accuracy, precision, recall and F1 score comparison for {f} features. Models: Attention(bl), MLP(ul), SVM(br), Tree(ur)}}
        \t\\label{{fig:{f}_radar}}
        \\end{{figure}}
        """

        f_path = out_dir / f'{f}_radar.tex'
        with open(f_path, 'w') as f:
            f.write(figure)


def ttest_table() -> None:
    metrics = [ACCURACY, PRECISION, RECALL, F1]
    data = load_results()
    params = get_params()

    data = mean_metrics_over_classes(data, [PRECISION, RECALL, F1])

    alpha = params['analysis']['alpha']
    features = params['data']['features']
    table_dir = Path(params['analysis']['tables'])
    table_dir.mkdir(exist_ok=True)
    for feature in features:
        f_data = data[data[FEATURES] == feature]

        metrics_better = ttest_analyze(f_data, metrics, alpha)
        generate_table(f_data, metrics_better, metrics, feature, table_dir)
