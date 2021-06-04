# %%
from multiprocessing import Pool
from pathlib import Path

import ffmpeg
import yaml
from tqdm import tqdm

from utils.utils import copy_dir_structure, get_files_and_destinations


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


def main() -> None:
    base_in_dir = Path(INPUT_DIR)
    base_out_dir = Path(OUTPUT_DIR)

    structure = copy_dir_structure(base_in_dir, base_out_dir)
    files = get_files_and_destinations(base_in_dir, base_out_dir, structure, '*.avi')

    with Pool() as p:
        r = list(tqdm(p.imap_unordered(extract_wrapper, files), total=len(files)))


if __name__ == '__main__':
    with open("params.yaml") as fd:
        params = yaml.safe_load(fd)

    INPUT_DIR = params['data']['datasets']['afew']['video']
    OUTPUT_DIR = params['data']['datasets']['afew']['wav']
    main()
