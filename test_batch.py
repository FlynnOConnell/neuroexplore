
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from data_utils import data_collection, file_handling, neural_signals, neuron, signals_food
from analysis import output
from nex.nexfile import Reader
from params import Params
from data_utils.file_handling import parse_filename

datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf')
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

info_df = pd.DataFrame(columns=['animal', 'date', 'neuron'])
means_df = pd.DataFrame(columns=params.stats_columns)
sems_df = pd.DataFrame(columns=params.stats_columns)

for file_path in datadir.glob('*.nex'):
    session_info = file_handling.parse_filename(file_path.name)
    __fileData = reader.ReadNexFile(str(file_path))
    nex = signals_food.EatingSignals(__fileData)
    session = [session_info.animal, session_info.date, nex.neuron]
    info_df.loc[len(info_df)] = session
    means_df = concat_series_to_dataframe(means_df, nex.means)
    sems_df = concat_series_to_dataframe(sems_df, nex.sems)

means_df.loc[-1] = ['mean' for x in means_df.columns]
means_df = means_df.sort_index().reset_index(drop=True)

sems_df.loc[-1] = ['sem' for x in sems_df.columns]
sems_df = sems_df.sort_index().reset_index(drop=True)

info_df.loc[-1] = ['info' for x in info_df.columns]
info_df = info_df.sort_index().reset_index(drop=True)

all_df = pd.concat([info_df, means_df, sems_df], axis=1)
# all_df.to_excel(r'C:\Users\Flynn\OneDrive\Desktop\temp\output2.xlsx', index=False)

if __name__ == '__main__':
    pass


# %%
# intervals = nex.intervals
#
# intervals_df = pd.DataFrame(columns=['event', 'start_time', 'end_time'])
# intervals_list = []
# for event, (start_times, end_times) in intervals.items():
#     if event not in ['AllFile', 'nospo']:
#         for start, end in zip(start_times, end_times):
#             intervals_list.append({'event': event, 'start_time': start, 'end_time': end})
#
# intervals_df = pd.DataFrame(intervals_list)
# neuron_df = pd.DataFrame({'timestamp': nex.neurostamps})
#
# def get_relative_timestamps(row, neuron_df):
#     start = row['start_time']
#     end = row['end_time']
#
#     mask = (neuron_df['timestamp'] >= start) & (neuron_df['timestamp'] <= end)
#     event_neurons = neuron_df[mask]['timestamp'] - start
#     relative_timestamps = event_neurons.to_list()
#
#     return np.asarray(relative_timestamps)
#
#
# intervals_df['relative_timestamps'] = intervals_df.apply(get_relative_timestamps, axis=1, neuron_df=neuron_df)
# intervals_df['num_spikes'] = intervals_df['relative_timestamps'].apply(len)
# intervals_df['duration'] = intervals_df['end_time'] - intervals_df['start_time']
# intervals_df = intervals_df[intervals_df['duration'] >= 1]
# intervals_df['spike_rate'] = intervals_df['num_spikes'] / intervals_df['duration']
#
# def bin_spikes(relative_timestamps, bin_size=1):
#     if hasattr(relative_timestamps, 'size') and relative_timestamps.size == 0:
#         return []
#
#     max_time = max(relative_timestamps)
#     num_bins = int(np.ceil(max_time / bin_size))
#     binned_spike_counts = np.zeros(num_bins)
#
#     for spike_time in relative_timestamps:
#         bin_index = int(spike_time // bin_size)
#         binned_spike_counts[bin_index] += 1
#
#     binned_spike_rates = binned_spike_counts / bin_size
#     return binned_spike_rates.tolist()
#
# bin_size = 1  # 1 second
# intervals_df['binned_spike_rates'] = intervals_df['relative_timestamps'].apply(bin_spikes, bin_size=bin_size)
# intervals_df['mean_spike_rate'] = intervals_df['binned_spike_rates'].apply(np.mean)
# intervals_df['median_spike_rate'] = intervals_df['binned_spike_rates'].apply(np.median)
# intervals_df['std_spike_rate'] = intervals_df['binned_spike_rates'].apply(np.std)
#
# event_summary_df = intervals_df.groupby('event')['mean_spike_rate'].agg(['mean', 'std']).reset_index()
#
# event_summary_df = intervals_df.groupby('event').agg({
#     'mean_spike_rate': ['mean', 'std'],
#     'median_spike_rate': 'mean'
# }).reset_index()



