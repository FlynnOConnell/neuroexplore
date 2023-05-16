import pandas as pd
import numpy as np
from pathlib import Path
import openpyxl
from params import Params
from data_utils import file_handling
params = Params()




class Output:
    """Class for outputting a single Signals data object to excel files"""
    def __init__(self, directory, data_dict: dict):
        self.directory = Path(directory)
        self.files = data_dict
        self.info_df = pd.DataFrame(columns=['animal', 'date', 'neuron'])

        self.means_df = pd.DataFrame(columns=params.stats_columns)
        self.sems_df = pd.DataFrame(columns=params.stats_columns)

    def output(self, file_path):
        self.file_outer.to_excel(file_path, index=False, sheet_name='file_stats',)
