import pickle as pkl
from pathlib import Path
from random import shuffle

import yaml


def load_data() -> dict[str, list[Path]]:
    all_files = list(DATA_DIR.glob('**/*.wav'))
    shuffle(all_files)

    data_dict = {}
    for file in all_files:
        emotion = file.stem.split('_')[-1]

        name = file.stem
        try:
            data_dict[emotion].append(name)
        except KeyError:
            data_dict[emotion] = [name]

    return data_dict


def create_chunks(data_dict: dict[str, list[Path]]) -> dict[str, list[list[Path]]]:
    chunk_dict = {}
    for emotion, files in data_dict.items():
        chunk_size = int(len(files) / FOLDS)
        chunks = []
        for i in range(FOLDS):
            chunk_start = i * chunk_size
            chunk_end = (i + 1) * chunk_size
            chunk = files[chunk_start:chunk_end]
            chunks.append(chunk)

        chunk_dict[emotion] = chunks

    return chunk_dict


def create_sets(chunks: list[list[Path]], fold: int) -> tuple[list[Path], list[Path]]:
    test_set = []
    train_set = []
    for i in range(FOLDS):
        if i == fold:
            test_set = chunks[i]
        else:
            train_set += chunks[i]

    return train_set, test_set


def split_data(chunk_dict: dict[str, list[list[Path]]]) -> list[dict[str, set[Path]]]:
    splits = []
    for i in range(FOLDS):
        all_train = []
        all_test = []
        for files in chunk_dict.values():
            train_set, test_set = create_sets(files, i)
            all_train += train_set
            all_test += test_set

        split_dict = {
            'train': set(all_train),
            'test': set(all_test),
        }
        splits.append(split_dict)

    return splits


if __name__ == '__main__':
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    FOLDS = params['train']['folds']
    DATA_DIR = Path(params['tess_wav'])
    OUTPUT_FILE = Path(params['tess_split'])

    data_dict = load_data()
    chunk_dict = create_chunks(data_dict)
    splits = split_data(chunk_dict)

    with open(OUTPUT_FILE, 'wb') as f:
        pkl.dump(splits, f, pkl.HIGHEST_PROTOCOL)
