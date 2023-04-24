
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# from data_utils import data_collection, file_handling, neural_signals, neuron, signals_food
from analysis import output
from nex.nexfile import Reader
from params import Params
from data_utils.file_handling import parse_filename
from data_utils import signals_food as sf
from data_utils import signals_stimuli as ss

datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\rs\SFN13_2018-12-11_RS_OBEX.nex')
reader = Reader(useNumpy=True)
params = Params()
def concat_series_to_dataframe(df, series):
    # Find the columns that are in the series but not in the dataframe
    new_columns = series.index.difference(df.columns)

    # Add the new columns to the dataframe with NaN values
    for col in new_columns:
        df[col] = np.nan

    # Convert the series to a dataframe and transpose it
    series_df = series.to_frame().T

    # Set the index of the new dataframe to the next row index of the original dataframe
    series_df.index = [df.index[-1] + 1] if len(df) > 0 else [0]

    # Concatenate the series dataframe to the original dataframe
    updated_df = pd.concat([df, series_df], axis=0)

    # Reorder the columns based on the original dataframe
    updated_df = updated_df[df.columns.union(new_columns)]

    return updated_df

# session_info = parse_filename(datadir.name)
__fileData = reader.ReadNexFile(str(datadir))
# nex = sf.EatingSignals(__fileData)

nex_rs = ss.StimuliSignals(__fileData, datadir.name)
nex_rs.get_allstats()

# for file_path in datadir.glob('*.nex'):
#
#     session_info = file_handling.parse_filename(file_path.name)
#     if (session_info.date == '2018-08-27') & (session_info.animal == 'SFN11'):
#         __fileData = reader.ReadNexFile(file_path)
#         nex = signals_food.Signals(__fileData)
#         means_df = concat_series_to_dataframe(means_df, nex.means)
#         sems_df = concat_series_to_dataframe(sems_df, nex.sems)
