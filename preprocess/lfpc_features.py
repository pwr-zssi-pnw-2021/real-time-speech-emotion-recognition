import pickle as pkl
from multiprocessing import Pool
from pathlib import Path

import librosa
import yaml
from tqdm import tqdm
from utils import copy_dir_structure, get_files_and_destinations


def extract_wrapper(file_and_dest: tuple[Path, Path]) -> None:
    file, destination = file_and_dest
    extract_lfpc_from_file(file, destination)


def extract_lfpc_from_file(file: Path, output_dir: Path) -> None:
    signal, _ = librosa.load(file)
    features = librosa.feature.lfpc(signal)

    output_file = output_dir / f'{file.stem}.pkl'

    with open(output_file, 'wb') as f:
        pkl.dump(features, f, pkl.HIGHEST_PROTOCOL)


def extract_lfpc_features(dataset_dir: Path, output_dir: Path, name: str) -> None:
    structure = copy_dir_structure(dataset_dir, output_dir)

    files_and_dest = get_files_and_destinations(
        dataset_dir, output_dir, structure, '*.wav'
    )

    with Pool() as p:
        r = list(
            tqdm(
                p.imap_unordered(extract_wrapper, files_and_dest),
                total=len(files_and_dest),
                desc=name,
            )
        )


if __name__ == '__main__':
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    AFEW_IN = Path(params['afew_wav'])
    AFEW_OUT = Path(params['afew_lfpc'])
    TESS_IN = Path(params['tess_wav'])
    TESS_OUT = Path(params['tess_lfpc'])

    extract_lfpc_features(AFEW_IN, AFEW_OUT, 'AFEW')
    extract_lfpc_features(TESS_IN, TESS_OUT, 'TESS')
