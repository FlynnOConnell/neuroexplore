"""
#file_helpers.py

Module(utils/file_helpers): File handling functions.
"""
from __future__ import annotations
import re
from collections import namedtuple
from typing import Dict, List
from pathlib import Path

def find_matching_files(directory: str | Path, match_string: str) -> Dict[str, List[str]]:
    """
    Find all files in the specified directory and its subdirectories that match the given pattern.

    Args:
        directory (str): The directory to search in.
        match_string (str): The pattern to match filenames against.

    Returns:
        Dict[str, List[str]]: A dictionary where the key is the subdirectory and the value is a list of matching filenames.
    """
    dir_path = Path(directory)
    matching_files = {}

    if not dir_path.exists() or not dir_path.is_dir():
        print(f"{directory} is not a valid directory.")
        return matching_files

    # Add current directory as key and files as values if not a subdir
    current_dir_files = [str(file.name) for file in dir_path.glob(match_string)]
    if current_dir_files:
        matching_files[dir_path.name] = current_dir_files

    for subdir in dir_path.iterdir():
        if subdir.is_dir():
            subdir_matching_files = [str(file.name) for file in subdir.glob(match_string)]
            matching_files[subdir.name] = subdir_matching_files

    return matching_files

def parse_filename(filename):
    # Define the regular expression pattern for the filename
    pattern = r'(?P<animal>[A-Za-z]{3,5}\d{1,2})_(?P<date>\d{4}-\d{2}-\d{2})_(?P<paradigm>RS|SF)'

    # Use the re.match function to find the matches in the filename
    match = re.match(pattern, filename)

    if match:
        # If the match is successful, extract the named groups
        animal = match.group('animal')
        date = match.group('date')
        paradigm = match.group('paradigm')

        # Define the namedtuple
        FileInfo = namedtuple('FileInfo', ['animal', 'date', 'paradigm'])

        # Create and return the named tuple with the extracted information
        return FileInfo(animal, date, paradigm)
    else:
        # If the match is not successful, return None
        return None

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
