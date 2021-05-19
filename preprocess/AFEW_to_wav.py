# %%
from multiprocessing import Pool
from pathlib import Path

import ffmpeg
from tqdm import tqdm

EMOTIONS = ['Angry', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sad', 'Surprise']
SETS = ['Train', 'Val']
INPUT_DIR = 'data/AFEW'
OUTPUT_DIR = 'data/AFEW_wav'


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

    files_to_extract = []
    for s in SETS:
        for e in EMOTIONS:
            in_dir = base_in_dir / s / e
            out_dir = base_out_dir / s / e
            for f in in_dir.glob('*.avi'):
                files_to_extract.append((f, out_dir))

    return files_to_extract


def create_output_structure() -> None:
    base_out_dir = Path(OUTPUT_DIR)

    base_out_dir.mkdir(exist_ok=True)
    for s in SETS:
        for e in EMOTIONS:
            out_dir = base_out_dir / s / e
            out_dir.mkdir(parents=True, exist_ok=True)


def main() -> None:
    files = get_files_and_destinations()

    create_output_structure()
    with Pool() as p:
        r = list(tqdm(p.imap_unordered(extract_wrapper, files), total=len(files)))


if __name__ == '__main__':
    main()
