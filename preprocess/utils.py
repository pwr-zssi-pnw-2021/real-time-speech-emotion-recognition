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
