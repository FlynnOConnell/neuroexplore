import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
import os
from data_reader import ReadData
from data_formatter import DataFormatter
from params import Params
from fourier import fft, stfft

def discrete_mean(arr):
    """Calculate the mean along the axis of a discrete array."""
    arr = np.asarray(arr)
    cumsum = np.cumsum(arr)
    cumsum[2:] = cumsum[2:] - cumsum[:-2]
    return np.asarray(cumsum[2 - 1:] / 2, dtype=int)

def rolling_average_with_std(arr, window_size=10):
    """
    Calculate the rolling average and standard deviation of an array. We use
    ddof=0 to ensure it's calculated as the population standard deviation.

    Parameters
    ----------
    :param arr: array to calculate rolling average and standard deviation of
    :param window_size: size of the window to calculate the rolling average and standard deviation

    Returns
    -------
    :return rolling_avg: array of rolling averages
    :return rolling_std: array of rolling standard deviations
    """
    rolling_avg_per_point = []
    rolling_std_per_point = []

    # Pad the input array with zeros to ensure it's divisible by window_size
    arr_padded = np.concatenate((arr, np.zeros(window_size - len(arr) % window_size)))

    # Calculate rolling average for the window
    for i in range(0, len(arr_padded), window_size):
        window = arr_padded[i:i + window_size]
        avg = np.mean(window)  # Calculate average for the window
        rolling_avg_per_point.extend([avg] * len(window))  # Extend the list with the window's average

    # Iterate over each value within the window and calculate standard deviation for that individual value
    for i in range(len(arr)):
        window_start = max(0, i - window_size + 1)
        window_end = min(i + 1, len(arr))
        window = arr[window_start:window_end]
        std = np.std(window, ddof=0)  # Calculate standard deviation for the window
        rolling_std_per_point.append(std)  # Append the standard deviation to the list

    return rolling_avg_per_point, rolling_std_per_point

# %%
params = Params()

nexdata = ReadData(params.filename, os.getcwd() + params.directory)
data = DataFormatter(nexdata.raw_data, params.well_events, params.eating_events, params.colors)
data_trials = data.trials

x=0
df_o = pd.DataFrame()
df1 = pd.DataFrame(columns=['isi', 'mid'])
for df in data.generate_trials_loose():
    x += 1
    if x == 1:
        df_o = df
        isi = np.diff(df.neuron)
        store = np.asarray(df['mid'].values[:-1], dtype=float)
        df1['mid'] = (store + store) / 2
        df1['isi'] = isi

avg, std = rolling_average_with_std(df1.isi)
df1['std'] = std
counts, bins = np.histogram(df1.isi, bins=np.arange(0, 0.2, 0.001))

fig, ax = plt.subplots()
ax.plot(df1.mid, df1.isi, color='black')
plt.show()

# discrete_mean(x_d)
# df1 = data.df_sorted
# df_e = df1[df1['event'].isin(eating_events)]
# df_s = df1[df1['event'].isin(['Spont', 'spont'])]
# df_w = df1[(df1.index <= df_e.index[0]) & (df1['event'].isin(well_events))]
#
# uni = data.df_sorted.loc[data.df_sorted.mid.drop_duplicates().index, ['neuron', 'color', 'counts', 'mid']]
#
# isi = np.diff(df_e.neuron)
# bins = np.arange(-.005, 0.2, 0.001)
# counts, _ = np.histogram(isi, bins)
# prob = counts / len(isi)
# fig, ax = plt.subplots()
# ax.bar(bins[:-1], prob, width=1e-3)
# ax.set_xlabel("ISI [s]")
# ax.set_ylabel("Frequency", fontweight='bold')
# ax.set_frame_on(False)
# plt.tight_layout()
# plt.show()
# for event, trial in data_trials.items():
#     if trial:
#         for trial_num, this_trial_df in data_trials[event].items():
#         # data_trials[event][trial]
#             single_histo(this_trial_df, 0)

# df_all_ev = pd.concat([df_spont, df_all_e, df_all_w], axis=0)
# df_all_ev = df_all_ev.drop('mid', axis=1).reset_index(drop=True)
# df_all_ev = df_all_ev.reset_index(drop=True)
#
# fig, ax = plt.subplots()
# ax.bar(df_all_ev.index, gaussian_filter(df_all_ev.counts, sigma=2), width=0.1, color=df_all_ev.color)
# ax.set_title("Everything", fontweight='bold')
# ax.axes.get_xaxis().set_visible(False)
# ax.set_ylabel("Frequency (hz)", fontweight='bold')
# ax.set_frame_on(False)
# # plt.tight_layout()
# plt.savefig(r'C:\Users\Flynn\Dropbox\Lab\fig')






