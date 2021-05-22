import pickle as pkl
from pathlib import Path
from random import shuffle

import yaml


def load_data() -> dict[str, list[Path]]:
    all_files = DATA_DIR.glob('*.pkl')
    data_dict = {}
    for file in all_files:
        emotion = file.stem.split('_')[-1]

        try:
            data_dict[emotion].append(file)
        except KeyError:
            data_dict[emotion] = [file]

    return data_dict


def split_data(data_dict: dict[str, list[Path]]) -> dict[str, set[Path]]:

    all_train_data = []
    all_test_data = []
    for _, files in data_dict.items():
        split_idx = int(len(files) * TEST_SPLIT_SIZE)

        shuffle(files)
        test_data = files[:split_idx]
        train_data = files[split_idx:]

        all_test_data += test_data
        all_train_data += train_data

    split_dict = {
        'train': set(all_train_data),
        'test': set(all_test_data),
    }

    return split_dict


if __name__ == '__main__':
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    TEST_SPLIT_SIZE = params['train']['test_split_size']
    DATA_DIR = Path(params['tess_mfcc'])
    OUTPUT_FILE = Path(params['tess_split'])

    data_dict = load_data()
    split_dict = split_data(data_dict)

    with open(OUTPUT_FILE, 'wb') as f:
        pkl.dump(split_dict, f, pkl.HIGHEST_PROTOCOL)
