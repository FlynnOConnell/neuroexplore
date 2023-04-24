from pathlib import Path
from collections import namedtuple
import numpy as np
import pandas as pd
from .file_handling import find_matching_files, parse_filename, get_nex
from . import signals_food as sf
from . import signals_stimuli as ss
from params import Params

class DataCollection:
    def __init__(self, directory: str | Path = ''):
        self.SignalData = None
        self.params = Params()
        self.stats = None
        self.info_df = pd.DataFrame(columns=['animal', 'date', '#timestamps'])
        self.means_df = pd.DataFrame(columns=self.params.stats_columns)
        self.sems_df = pd.DataFrame(columns=self.params.stats_columns)

        if not directory:
            self.directory = Path(self.params.directory)
        else:
            self.directory = Path(directory)
        self.data_files = find_matching_files(self.directory, '*.nex*')
        self.files = {paradigm: [] for paradigm in self.data_files if paradigm in ['sf', 'rs']}
        self.errors = {}

    def get_data(self, paradigm='sf', num_files=None):
        if paradigm not in self.data_files:
            raise ValueError(f'Paradigm {paradigm} not found.')
        if num_files:
            self.data_files[paradigm] = self.data_files[paradigm][:num_files]
        for file in self.data_files[paradigm]:
            try:
                nexfile = get_nex(self.directory / paradigm / file)
                if paradigm == 'sf':
                    data = sf.EatingSignals(nexfile)
                elif paradigm == 'rs':
                    data = ss.StimuliSignals(nexfile, file)
                else:
                    raise ValueError(f'Paradigm {paradigm} not found.')
                self.files[paradigm].append({file: data})
            except Exception as e:
                self.errors[file] = e

    @staticmethod
    def concat_dataframe_to_dataframe(df1, df2):
        # Find the columns that are in df2 but not in self.means_df
        new_columns = df2.columns.difference(df1.columns)

        # Add the new columns to self.means_df with NaN values
        for col in new_columns:
            df1[col] = np.nan

        # Find the columns that are in self.means_df but not in df2
        missing_columns = df1.columns.difference(df2.columns)

        # Add the missing columns to df2 with NaN values
        for col in missing_columns:
            df2[col] = np.nan

        # Concatenate df2 to self.means_df along the rows (axis=0)
        updated_df = pd.concat([df1, df2], axis=0)

        # Reorder the columns based on the original self.means_df
        updated_df = updated_df[df1.columns]

        # Reset the index
        updated_df.reset_index(drop=True, inplace=True)

        return updated_df

    def get_stats(self, paradigm='sf'):
        for file in self.files[paradigm]:
            for filename, data in file.items():
                session_info = parse_filename(filename)
                session = [session_info.animal, session_info.date]
                for neuron in data.neurons:
                    self.info_df.loc[len(self.info_df)] = session + [data.num_ts[neuron]]
                self.means_df = self.concat_dataframe_to_dataframe(self.means_df, data.means)
                self.sems_df = self.concat_dataframe_to_dataframe(self.sems_df, data.sems)
                if self.means_df.shape != self.sems_df.shape:
                    raise ValueError('Means and SEMs dataframes are not the same size.')
        self.stats = pd.concat([self.info_df, self.means_df, self.sems_df], axis=1)

    def output(self, outpath):
        self.stats.to_excel(outpath, index=False, sheet_name='stats')