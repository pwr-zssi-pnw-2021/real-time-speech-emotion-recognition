# %%
from multiprocessing import Pool
from pathlib import Path

import ffmpeg
import yaml
from tqdm import tqdm

from utils import STRUCTURE, copy_dir_structure


def extract_audio(file: Path, output_dir: Path) -> None:
    new_name = name_avi_to_wav(file.name)
    output_path = output_dir / new_name

    stream = ffmpeg.input(file)
    stream.audio.output(filename=output_path, loglevel='error').run()


def extract_wrapper(data: tuple[Path, Path]) -> None:
    file, output_dir = data
    extract_audio(file, output_dir)


def name_avi_to_wav(name: str) -> str:
    return f'{name.split(".")[0]}.wav'


def get_files_and_destinations() -> list[tuple[Path, Path]]:
    base_in_dir = Path(INPUT_DIR)
    base_out_dir = Path(OUTPUT_DIR)

    structure = copy_dir_structure(base_in_dir, base_out_dir)
    files_to_extract = _rec_get_files_and_destinations(
        base_in_dir, base_out_dir, structure
    )

    return files_to_extract


def _rec_get_files_and_destinations(
    base_in_dir: Path, base_out_dir: Path, structure: STRUCTURE
) -> list[tuple[Path, Path]]:
    files_to_extract = []
    for dname, children in structure.items():
        in_dir = base_in_dir / dname
        out_dir = base_out_dir / dname
        for f in in_dir.glob('*.avi'):
            files_to_extract.append((f, out_dir))
        child_list = _rec_get_files_and_destinations(in_dir, out_dir, children)
        files_to_extract += child_list
    return files_to_extract


def main() -> None:
    files = get_files_and_destinations()

    with Pool() as p:
        r = list(tqdm(p.imap_unordered(extract_wrapper, files), total=len(files)))


if __name__ == '__main__':
    with open("params.yaml", 'r') as fd:
        params = yaml.safe_load(fd)

    INPUT_DIR = params['afew_raw']
    OUTPUT_DIR = params['afew_wav']
    main()
