import pickle as pkl
from pathlib import Path

from utils.utils import get_params

if __name__ == '__main__':
    params = get_params()

    index_dir = Path(params['data']['index_dir'])

    global_index_dir = Path(params['data']['global_index_dir'])
    global_index_dir.mkdir(exist_ok=True)

    features = params['data']['features']
    index_dic = {
        f: {
            'paths': [],
            'classes': [],
        }
        for f in features
    }

    for path in index_dir.glob('*.index'):
        ft = path.stem.split('_')[-1]
        with open(path, 'rb') as f:
            paths, classes = pkl.load(f)

        index_dic[ft]['paths'].extend(paths)
        index_dic[ft]['classes'].extend(classes)

    for f_name, idx in index_dic.items():
        idx_path = global_index_dir / f'{f_name}.index'
        with open(idx_path, 'wb') as f:
            pkl.dump(
                (
                    idx['paths'],
                    idx['classes'],
                ),
                f,
                pkl.HIGHEST_PROTOCOL,
            )
