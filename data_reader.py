"""
#file_helpers.py

Module(utils/file_helpers): File handling data-container class to keep all file-related
data.
"""
from __future__ import annotations

from nex import nexfile

from helpers import funcs, ax_helpers
import logging
from collections import namedtuple
from pathlib import Path
from typing import Optional

import pandas as pd

from helpers import funcs

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger.setLevel(logging.DEBUG)


class ReadData:
    """
    File handler. From directory, return genorator looping through files.

    Parameters:
    ___________
    animal: str
        - Animal ID, letter + number combo
    date: str
        - Session date, all_numeric
    _directory: str
        - Path as str, contains /
    _tracename: Optional[str]
        - Pattern matching to apply for trace files
    _eventname: Optional[str]
        - Pattern matching to apply for event files

    """

    def __init__(
            self,
            filename,
            directory,
    ) -> None:

        # TODO: glob for .csv and .xlxs
        self.file = filename
        self._directory: Path = Path(directory)
        self.reader = nexfile.Reader(useNumpy=True)
        self.raw_data = self.get_filedata()
        self.color_dict: namedtuple
        self._make_dirs()

    @property
    def directory(self) -> Path:
        return self._directory

    @directory.setter
    def directory(self, new_dir: str) -> None:
        self._directory: str = new_dir

    @directory.setter
    def directory(self, new_dir: str) -> None:
        self._directory: str = new_dir

    def get_files(self) -> list[Path]:
        return [Path(p) for p in self.directory.glob(f"*{self.file}*")]

    def get_filedata(self) -> dict:
        nexfiles: list[Path] = self.get_files()
        if not nexfiles:
            files = self.search_files()
            raise FileNotFoundError(
                f'No files in {self.directory} matching "{self.file}"'
                f"Files found: {files}"
            )

        if len(nexfiles) > 1:
            logging.info(
                f'Multiple trace-files found in {self.directory} matching "'
                f'{self.file}":'
            )
        for file in nexfiles:
            logging.info(f"{file.stem}")
        p: Path = nexfiles[0]
        pa = p.resolve()
        return self.reader.ReadNexFile(pa)

    def unique_path(self, filename) -> Path:
        counter = 0
        while True:
            counter += 1
            path = self.directory / filename
            if not path.exists():
                return path

    def _make_dirs(self) -> None:
        self.directory.parents[0].mkdir(parents=True, exist_ok=True)
        return None

    def get_cwd(self) -> str:
        return str(self._directory.cwd())

    def get_home_dir(self) -> str:
        return str(self._directory.home())

    def tree(self) -> None:
        print(f"-|{self._directory}")
        for path in sorted(self.directory.rglob("[!.]*")):  # exclude .files
            depth = len(path.relative_to(self.directory).parts)
            spacer = "    " * depth
            print(f"{spacer}-|{path.name}")
            return None

    def search_files(self):
        dirpath = self._directory
        if not dirpath.is_dir():
            raise IsADirectoryError(f'{dirpath} is not a valid directory')
        file_list = []
        for x in dirpath.iterdir():
            if x.is_file():
                file_list.append(x)
        return file_list


