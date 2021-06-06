import numpy as np
from scipy.stats import ttest_rel


def ttest(data: np.ndarray, alpha: float = 0.05) -> np.ndarray:
    ds = len(data)
    t_stats = np.zeros((ds, ds))
    p_vals = np.zeros((ds, ds))
    for i in range(ds):
        for j in range(ds):
            t_stats[i, j], p_vals[i, j] = ttest_rel(data[i], data[j])
    bin_t_stats = t_stats > 0
    bin_significance = p_vals <= alpha

    difference = bin_t_stats * bin_significance
    return difference.astype(int)
