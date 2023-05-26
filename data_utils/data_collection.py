from pathlib import Path
from collections.abc import Iterable

import sys
import traceback
import numpy as np
import pandas as pd
from .file_handling import find_matching_files, parse_filename, get_nex

from . import signals_food as sf
from . import signals_stimuli as ss
from params import Params

def concat_dataframe_to_dataframe(df1, df2):
    """
    Concatenates two pandas DataFrames along the rows, adding NaN for any missing columns in either DataFrame.

    The function first finds and adds any new columns that are present in `df2` but not in `df1`. It then finds
    and adds any missing columns that are present in `df1` but not in `df2`. The dataframes are then concatenated
    along the rows (axis=0). The resulting DataFrame has the same column order as `df1` and the index is reset.

    Args:
        df1 (pandas.DataFrame): The first DataFrame.
        df2 (pandas.DataFrame): The second DataFrame to be concatenated to the first DataFrame.

    Returns:
        pandas.DataFrame: The resulting concatenated DataFrame.

    Examples:
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df2 = pd.DataFrame({'B': [5, 6], 'C': [7, 8]})
        >>> concat_dataframe_to_dataframe(df1, df2)
           A  B   C
        0  1  3 NaN
        1  2  4 NaN
        2 NaN  5 7.0
        3 NaN  6 8.0
    """
    # Find the columns that are in df2 but not in df1
    new_columns = df2.columns.difference(df1.columns)

    # Add the new columns to df1 with NaN values
    for col in new_columns:
        df1[col] = np.nan

    # Find the columns that are in df1 but not in df2
    missing_columns = df1.columns.difference(df2.columns)

    # Add the missing columns to df2 with NaN values
    for col in missing_columns:
        df2[col] = np.nan

    updated_df = pd.concat([df1, df2], axis=0)

    # Reorder the columns based on the original df1
    updated_df = updated_df[df1.columns]
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
        Class for collecting data from multiple Signals data objects.
    """
    def __init__(self, directory: str | Path, recursive: bool = False, opto: bool = False):

        self.files = {}
        self.params = Params()
        self.opto = opto
        self.means = pd.DataFrame(columns=self.params.stats_columns)
        self.sems = pd.DataFrame(columns=self.params.stats_columns)
        self.std = pd.DataFrame(columns=self.params.stats_columns)
        self.session_stats = pd.DataFrame()

        self.stats = None
        self.info_df = pd.DataFrame(columns=['animal', 'date', '#waveforms'])

        self.directory = Path(directory)
        self.data_files = find_matching_files(self.directory, '*.nex*', recursive=recursive)
        self.errors = {}

    def __repr__(self):
        return f'{len(self.data_files)} files'

    @property
    def num_files(self):
        return len(self.files)

    # for multiple files
    def get_data(self, paradigm='SF', num_files=None, functions_to_run=None, exclude=None):
        """
        Instantiate Signals class for given paradigm. If num_files is not None, only the first num_files will be used.
        Sorted by paradigm, then filename.
        """
        if exclude is None:
            exclude = []
        if num_files:
            self.data_files = self.data_files[:num_files]
        for file in self.data_files:
            if file not in exclude:
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
            if file in filenames:
                try:
                    nexfile = get_nex(self.directory / file)
                    if paradigm in ['sf', 'SF']:
                        data = sf.EatingSignals(nexfile, file, self.opto,)
                        self.files[file] = data
                    elif paradigm in ['rs', 'RS']:
                        data = ss.StimuliSignals(nexfile, file)
                        if functions_to_run:
                            data.run_stats(functions_to_run)
                        self.files[file] = data
                    else:
                        raise ValueError(f'Paradigm {paradigm} not found.')
                except Exception as e:
                    exc_type, exc_value, exc_traceback = sys.exc_info()
                    tb_str = traceback.format_exception(exc_type, exc_value, exc_traceback)
                    tb_lines = ''.join(tb_str)
                    self.errors[file] = f"{e}\n{tb_lines}"
        return nexdata

    def get_session_stats(self):
        for filename, data in self.files.items():
            self.session_stats = concat_dataframe_to_dataframe(self.session_stats, data.trial_stats)

    def get_eating_stats(self):
        for filename, data in self.files.items():
            session_info = parse_filename(filename)
            session = [session_info.animal, session_info.date]
            for neuron in data.neurons.keys():
                # one row per neuron, fill timestamps
                self.info_df.loc[len(self.info_df)] = session + [data.waveforms[neuron]]
            self.means = concat_dataframe_to_dataframe(self.means, data.means)
            self.sems = concat_dataframe_to_dataframe(self.sems, data.sems)
            self.std = concat_dataframe_to_dataframe(self.std, data.stds)

            if self.means.shape != self.sems.shape:
                raise ValueError(f'Means and SEMs dataframes are not the same size:'
                                 f' \n {filename} \n {self.means.shape} \n {self.sems.shape}')
        self.stats = pd.concat([self.info_df, self.means, self.sems, self.std], axis=1)

    @staticmethod
    def output(data, output_dir, filename='stats', sheet_name='Sheet1',):
        # import here to avoid circular import and for clarity
        from analysis import output
        file = output.SaveSignals(data, output_dir)
        file.save_file(filename, sheetname=sheet_name)
