from pathlib import Path
from collections.abc import Iterable

import sys
import traceback
import numpy as np
import pandas as pd
from data_utils.file_handling import find_matching_files, parse_filename, get_nex

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


class eating_statistics:
    def __init__(self, files    # list of files to be analyzed
                 ):
        self.files = files
        self.get_session_stats = self.get_session_stats()
        self.eating_stats = self.get_eating_stats()

    def __repr__(self):
        return f'{len(self.files)} files'

    def get_session_stats(self):
        session_stats = pd.DataFrame()
        for filename, data in self.files.items():
            session_stats = concat_dataframe_to_dataframe(session_stats, data.trial_stats)
        return session_stats

    def get_eating_stats(self):
        info_df = pd.DataFrame(columns=['animal', 'date'])
        means = pd.DataFrame()
        sems = pd.DataFrame()
        stds = pd.DataFrame()
        for filename, data in self.files.items():
            session_info = parse_filename(filename)
            session = [session_info.animal, session_info.date]
            for neuron in data.neurons.keys():
                # one row per neuron, fill timestamps
                info_df.loc[len(info_df)] = session
            means = concat_dataframe_to_dataframe(means, data.means).reset_index(drop=True)
            sems = concat_dataframe_to_dataframe(sems, data.sems).reset_index(drop=True)
            stds = concat_dataframe_to_dataframe(stds, data.stds).reset_index(drop=True)

            if means.shape != sems.shape:
                raise ValueError(f'Means and SEMs dataframes are not the same size:'
                                 f' \n {filename} \n {means.shape} \n {sems.shape}')
        return pd.concat([info_df, means, sems, stds], axis=1)
