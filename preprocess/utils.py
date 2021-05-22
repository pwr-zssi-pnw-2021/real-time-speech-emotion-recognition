from pathlib import Path

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
