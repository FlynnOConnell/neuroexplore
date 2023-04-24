import pandas as pd
from pathlib import Path
import openpyxl

class Output:
    def __init__(self, directory, data_dict: dict):
        self.directory = Path(directory)
        self.files = data_dict
        self.file_outer = None
        self.trials_outer = None

    def concat_df(self):
        self.trials_outer = pd.DataFrame()
        self.file_outer = pd.DataFrame()
        for day, data in self.files.items():
            # trial_df = data.stats_trials.reset_index(drop=True)
            file_df = data.stats_file.reset_index(drop=True)
            # self.trials_outer = pd.concat([self.trials_outer, trial_df], axis=0, ignore_index=True)
            self.file_outer = pd.concat([self.file_outer, file_df], axis=0,  ignore_index=True)

    def output(self, file_path):
        self.file_outer.to_excel(file_path,
                                 index=False,
                                 sheet_name='file_stats',)
