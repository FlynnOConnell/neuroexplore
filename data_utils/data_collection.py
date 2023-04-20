from pathlib import Path
from .file_handling import find_matching_files, parse_filename
from . import neuron as dh
from params import Params

class DataCollection:
    def __init__(self, directory: str | Path = ''):
        self.params = Params()
        if not directory:
            self.directory = Path(self.params.directory)
        else:
            self.directory = Path(directory)
        self.data_files = find_matching_files(self.directory, '*.nex*')
        self.files = {}
        self.errors = {}

    def get_data_sf(self, num_files=None):
        if num_files:
            self.data_files['sf'] = self.data_files['sf'][:num_files]
        for file in self.data_files['sf']:
            try:
                data = dh.Neuron(self.directory / 'sf' / file)
                self.files[data.file_name] = data
            except Exception as e:
                self.errors[file] = e
