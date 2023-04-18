"""
#file_helpers.py

Module(utils/file_helpers): File handling functions.
"""
from __future__ import annotations
from nex import nexfile
from pathlib import Path

def find_matching_files(directory: str | Path, match_string: str | Path) -> list[Path]:
    """
    Find all files in the specified directory that match the given pattern.

    Args:
        directory (str): The directory to search in.
        match_string (str): The pattern to match filenames against.

    Returns:
        list[Path]: A list of matching file paths as pathlib.Path.
    """
    dir_path = Path(directory)
    matching_files = []

    if not dir_path.exists() or not dir_path.is_dir():
        print(f"{directory} is not a valid directory.")
        return matching_files

    for file in dir_path.glob(match_string):
        if file.is_file():
            matching_files.append(Path(file))
    return matching_files

def get_nexfiles(files: list[Path]) -> list[dict]:
    if not files:
        raise FileNotFoundError(
            f'No nex files found.'
        )
    return [nexfile.Reader(useNumpy=True).ReadNexFile(filename.resolve()) for filename in files]

def unique_path(directory, filename) -> Path:
    counter = 0
    while True:
        counter += 1
        path = directory / filename
        if not path.exists():
            return path

def tree(directory) -> None:
    print(f"-|{directory}")
    for path in sorted(directory.rglob("[!.]*")):  # exclude .files
        depth = len(path.relative_to(directory).parts)
        spacer = "    " * depth
        print(f"{spacer}-|{path.name}")
        return None
