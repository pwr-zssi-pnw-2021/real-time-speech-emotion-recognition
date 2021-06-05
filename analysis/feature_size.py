# %%
import pickle as pkl
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import yaml


def load_lens(features: str) -> tuple[np.ndarray, np.ndarray]:
    with open('../params.yaml') as f:
        params = yaml.safe_load(f)

    index_dir = '..' / Path(params['data']['global_index_dir'])
    index_file = index_dir / f'{features}.index'

    with open(index_file, 'rb') as f:
        files, _ = pkl.load(f)

    f_lens = []
    d_lens = []
    for file in files:
        file = '..' / file
        with open(file, 'rb') as f:
            d = pkl.load(f)
        d_lens.append(d.shape[0])
        f_lens.append(d.shape[1])

    return np.array(d_lens), np.array(f_lens)


# %%
lens, size = load_lens('sc')
# %%
# Percentile
p = np.percentile(size, 1)
plt.hist(size, bins=50)
plt.vlines(p, ymin=0, ymax=400, colors='r')
plt.show()
print(p)
# %%
# Max size
lens.max()
# %%
plt.hist(lens, bins=50)
plt.show()
# %%
