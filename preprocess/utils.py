import functools
import pickle as pkl
from multiprocessing import Pool
from pathlib import Path
from typing import Callable

import librosa
import numpy as np
import yaml
from spafe.features.lfcc import lfcc
from spafe.features.lpc import lpcc
from tqdm import tqdm

STRUCTURE = dict[str, 'STRUCTURE']


def copy_dir_structure(source: Path, destination: Path) -> STRUCTURE:
    destination.mkdir(exist_ok=True)
    struct = _rec_copy_dir_structure(source, destination)

    return struct


def _rec_copy_dir_structure(source: Path, destination: Path) -> STRUCTURE:
    source_dirs = [d for d in source.glob('*') if d.is_dir()]
    struct = {}
    for sdir in source_dirs:
        new_dir = destination / sdir.name
        new_dir.mkdir(exist_ok=True)

        struct[sdir.name] = _rec_copy_dir_structure(sdir, new_dir)

    return struct


def get_files_and_destinations(
    base_in_dir: Path,
    base_out_dir: Path,
    structure: STRUCTURE,
    pattern: str,
) -> list[tuple[Path, Path]]:
    files_and_dest = []
    for dname, children in structure.items():
        in_dir = base_in_dir / dname
        out_dir = base_out_dir / dname
        for f in in_dir.glob(pattern):
            files_and_dest.append((f, out_dir))
        child_list = get_files_and_destinations(in_dir, out_dir, children, pattern)
        files_and_dest += child_list
    return files_and_dest


def get_params() -> dict:
    with open('params.yaml') as f:
        params = yaml.safe_load(f)

    return params


def get_mfcc_features(file: Path) -> np.ndarray:
    signal, sr = librosa.load(file)
    features = librosa.feature.mfcc(signal, sr)

    return features


def get_lpcc_features(file: Path) -> np.ndarray:
    signal, sr = librosa.load(file)
    features = lpcc(signal, sr)

    return features


def get_lfcc_features(file: Path) -> np.ndarray:
    signal, sr = librosa.load(file)
    features = lfcc(signal, sr)

    return features


def generate_extractor(
    label_extractor: Callable, features_extractor: Callable
) -> Callable:
    def extractor(file: Path, output_dir: Path) -> tuple[Path, int]:
        label = label_extractor(file)
        features = features_extractor(file)
        out_file = save_features(features, output_dir, file.stem)

        return out_file, label

    return extractor


def get_tess_class(file: Path) -> int:
    emotions = get_params()['data']['emotions']

    dir_name = file.parent.name
    emotion = dir_name.split('_')[-1]
    emotion = emotion.lower()

    for i, e in enumerate(emotions):
        if emotion in e:
            return i

    raise ValueError(f'Emotion not found in the list: {emotion}')


def get_afew_class(file: Path) -> int:
    emotions = get_params()['data']['emotions']

    dir_name = file.parent.name
    emotion = dir_name.lower()

    for i, e in enumerate(emotions):
        if emotion in e:
            return i

    raise ValueError(f'Emotion not found in the list: {emotion}')


def save_features(features: np.ndarray, output_dir: Path, name: str) -> Path:
    output_file = output_dir / f'{name}.pkl'
    if output_file.exists():
        raise ValueError(f'File already exists: {output_file}')

    with open(output_file, 'wb') as f:
        pkl.dump(features, f, pkl.HIGHEST_PROTOCOL)

    return output_file


def create_index_file(index: list[tuple[Path, int]], path: Path) -> None:
    path.parent.mkdir(exist_ok=True)

    index_split = list(zip(*index))
    with open(path, 'wb') as f:
        pkl.dump(index_split, f, pkl.HIGHEST_PROTOCOL)


def mp_wrapper(func: Callable) -> Callable:
    @functools.wraps(func)
    def wrapper(args: tuple[Path, Path]) -> tuple[Path, int]:
        file = args[0]
        output_dir = args[1]

        return func(file, output_dir)

    return wrapper


def extract(
    data_path: str,
    output_path: str,
    index_path: str,
    extract_func: Callable,
):
    data_dir = Path(data_path)
    out_dir = Path(output_path)
    index_file = Path(index_path)

    out_dir.mkdir(exist_ok=True)
    in_data = [(f, out_dir) for f in data_dir.glob('*.wav')]
    with Pool() as p:
        index = list(
            tqdm(
                p.imap_unordered(extract_func, in_data),
                total=len(in_data),
            )
        )

    create_index_file(index, index_file)


AFEW = 'afew'
TESS = 'tess'

DATASETS = [
    AFEW,
    TESS,
]

MFCC = 'mfcc'
LPCC = 'lpcc'
LFCC = 'lfcc'

FEATURES = [
    MFCC,
    LPCC,
    LFCC,
]


FEATURE_EXTRACTOR_LOOKUP = {
    MFCC: get_mfcc_features,
    LPCC: get_lpcc_features,
    LFCC: get_lfcc_features,
}

CLASS_EXTRACTOR_LOOKUP = {
    TESS: get_tess_class,
    AFEW: get_afew_class,
}
