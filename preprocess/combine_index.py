import pickle as pkl
from pathlib import Path

from utils import get_params

if __name__ == '__main__':
    params = get_params()

    index_dir = Path(params['data']['index_dir'])
    global_index = Path(params['data']['global_index'])

    all_paths = []
    all_classes = []
    for path in index_dir.glob('*'):
        with open(path, 'rb') as f:
            paths, classes = pkl.load(f)
        all_paths.extend(paths)
        all_classes.extend(classes)

    with open(global_index, 'wb') as f:
        pkl.dump((all_paths, all_classes), f, pkl.HIGHEST_PROTOCOL)
