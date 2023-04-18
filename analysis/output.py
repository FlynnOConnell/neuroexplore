import os
import pandas as pd
from pathlib import Path


class Output:
    def __init__(self, directory):
        self.directory = Path(directory)
        self.files = self.get_data_files()

    @staticmethod
    def to_excel(self, data, file_path):
        df = pd.DataFrame(data)
        df.to_excel(file_path, index=False)

    def get_data_files(self):
        return [str(file_path.resolve()) for file_path in self.directory.glob('*.nex*')]

    def get_data(self):
        return [pd.read_excel(file_path) for file_path in self.files]


if __name__ == "__main__":
    from data_utils import data_holder as data
    from data_utils.file_loaders import find_matching_files, get_nexfiles
    # data = data.Data()
    datadir = 'C:\\repos\\neuroexplore\\data'
    files = find_matching_files(datadir, '*.nex*')
    nexfiles = get_nexfiles(files)
    data = data.Data(nexfiles[0])
    # output = Output(datadir)