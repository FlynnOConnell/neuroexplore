from pathlib import Path
from collections.abc import Iterable

import openpyxl
from openpyxl import load_workbook
import numpy as np
import sys
import traceback
import pandas as pd
from .file_handling import find_matching_files, parse_filename, get_nex
from . import signals_food as sf
from . import signals_stimuli as ss
from params import Params

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

def concat_series_to_dataframe(df, series):
    # Find the columns that are in the series but not in the dataframe
    new_columns = series.index.difference(df.columns)
    for col in new_columns:
        df[col] = np.nan
    series_df = series.to_frame().T
    # Set the index of the new dataframe to the next row index of the original dataframe
    series_df.index = [df.index[-1] + 1] if len(df) > 0 else [0]
    updated_df = pd.concat([df, series_df], axis=0)

    # Reorder the columns based on the original dataframe
    updated_df = updated_df[df.columns.union(new_columns)]

    return updated_df

class DataCollection:
    """
        Class for collecting data from multiple Signals data objects
        Files is a dictionary of lists of files, where the keys are the paradigms or subdirectories
        in given directory, and the values are lists of files in those subdirectories.
    """
    def __init__(self, directory: str | Path = '', recursive: bool = False, opto: bool = False):

        self.files = {}
        self.params = Params()
        self.opto = opto
        self.means_df = pd.DataFrame(columns=self.params.opto_events)
        self.sems_df = pd.DataFrame(columns=self.params.opto_events)
        self.trials_df = pd.DataFrame(columns=self.params.opto_events)
        self.mean_time_df = pd.DataFrame(columns=self.params.opto_events)
        self.sem_time_df = pd.DataFrame(columns=self.params.opto_events)
        self.stats = None

        self.info_df = pd.DataFrame(columns=['animal', 'date', '#waveforms'])

        if not directory:
            self.directory = Path(self.params.directory)
        else:
            self.directory = Path(directory)
        self.data_files = find_matching_files(self.directory, '*.nex*', recursive=recursive)
        self.errors = {}

    def __repr__(self):
        return f'{len(self.data_files)} files'

    # for multiple files
    def get_data(self, paradigm='SF', num_files=None, functions_to_run=None,):
        """
        Instantiate Signals class for given paradigm. If num_files is not None, only the first num_files will be used.
        Sorted by paradigm, then filename.
        """
        if num_files:
            self.data_files = self.data_files[:num_files]
        for file in self.data_files:
            try:
                nexfile = get_nex(self.directory / file)
                if paradigm in ['sf', 'SF']:
                    data = sf.EatingSignals(nexfile, file, self.opto,)
                elif paradigm in  ['rs', 'RS']:
                    data = ss.StimuliSignals(nexfile, file)
                    if functions_to_run:
                        data.run_stats(functions_to_run)
                else:
                    raise ValueError(f'Paradigm {paradigm} not found.')
                self.files[file] = data
            except Exception as e:
                exc_type, exc_value, exc_traceback = sys.exc_info()
                tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                tb_lines = ''.join(tb_str)
                self.errors[file] = f"{e}\n{tb_lines}"

    # for single file
    def get_data_by_filename(self, filenames, paradigm='sf', functions_to_run=None,):
        """
        Instantiate Signals class for given filename(s).
        filenames can be a single string or an iterable of strings.
        """
        nexdata = {}
        if not isinstance(filenames, Iterable):
            filenames = [filenames]
        for file in self.data_files:
            for name in filenames:
                try:
                    nexfile = get_nex(self.directory / name)
                    data = ss.StimuliSignals(nexfile, file)
                    if functions_to_run:
                        data.run_stats(functions_to_run)
                    nexdata[name] = data
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    tb_lines = ''.join(tb_str)
                    self.errors[file] = f"{e}\n{tb_lines}"
        return nexdata


    def get_eating_stats(self):
        for filename, data in self.files.items():
            session_info = parse_filename(filename)
            session = [session_info.animal, session_info.date]
            for neuron in data.neurons.keys():
                # one row per neuron, fill timestamps
                self.info_df.loc[len(self.info_df)] = session + [data.num_ts[neuron]]
            self.means_df = concat_dataframe_to_dataframe(self.means_df, data.mean_df)
            self.sems_df = concat_dataframe_to_dataframe(self.sems_df, data.sem_df)
            self.trials_df = concat_dataframe_to_dataframe(self.trials_df, data.trials_df)
            self.mean_time_df = concat_dataframe_to_dataframe(self.mean_time_df, data.mean_time_df)
            self.sem_time_df = concat_dataframe_to_dataframe(self.sem_time_df, data.mean_sem_df)
            if self.means_df.shape != self.sems_df.shape:
                raise ValueError('Means and SEMs dataframes are not the same size.')
        self.stats = pd.concat([self.info_df, self.means_df, self.sems_df,
                                self.trials_df, self.mean_time_df,  self.sem_time_df], axis=1)

