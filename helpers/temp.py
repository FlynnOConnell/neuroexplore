
# %% PLOT TRIALS -----------------------------------------------------------------

# x=0
# well_times, eating_times, spont_times = [], [], []
# df1 = pd.DataFrame(columns=['isi', 'mid'])
# for df, wt, et, st in data.generate_trials_loose():
#
#     plot.isi_histogram(df.neuron, df.color)
#     plot.trial_histogram(df)
#     avg, std = calculate_window_stats(np.diff(df.neuron))
#     fig, ax = plt.subplots()
#     ax.plot(df.reset_index(drop=True).index[:-1], std, color='black')
#     ax.set_title("St. Dev. of Sequential ISIs \n"
#                  "(sliding window of 10 intervals)", fontweight='bold')
#     ax.set_xlabel("ISI number", fontweight='bold')
#     ax.set_ylabel("st. dev. (# of spikes)", fontweight='bold')
#     ax.set_frame_on(False)
#     ax.axvspan(
#         et[0],
#         et[-1],
#         alpha=0.05,
#         color='red'
#     )
#     ax.axvspan(
#         wt[0],
#         wt[-1],
#         alpha=0.05,
#         color='blue'
#     )
#     if st.size > 0:
#         ax.axvspan(
#             st[0],
#             st[-1],
#             alpha=0.05,
#             color='gray'
#         )
#     plt.tight_layout()
#     plt.show()

# %% PLOT ISI'S FOR ALL TRIALS ------------------------------------------------

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

# %% DATA TO MATLAB ----------------------------------------------------------

# final = {}
#
# for k, v in spike_times_dict.items(): # k = e_BR, v = {3: array, 4: array}
#     for outer, inner in v.items(): # outer = 3, inner = array
#         for k2, v2 in inner.items():
#
#         mat_structs.__setattr__(outer, inner)
#     final[k] = mat_structs
#

# scipy.io.savemat(savepath + 'from_python.mat', final)


from __future__ import annotations
import warnings
from collections import namedtuple
import numpy as np
import scipy.stats as stats
import pandas as pd
from params import Params
warnings.simplefilter(action='ignore', category=FutureWarning)

class EatingSignals:
    def __init__(self, nexdata: dict) -> None:
        self.nex = nexdata
        self.ensemble = False
        self.params = Params()
        self.neurons: dict = {}
        self.neurostamps = []
        self.intervals = {}
        self.__populate()
        self.intervals_df = self.build_intervals_df()

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.nex["Header"]["Filename"]})'

    def build_intervals_df(self):
        for neuron, timestamps in self.neurons.items():
            neur = pd.DataFrame()
            intervals_df = self.get_intervals_df()
            intervals_df['relative_timestamps'] = intervals_df.apply(
                self.get_relative_timestamps, axis=1, neuron_df=self.__neuron_df)
            intervals_df['binned_spike_rates'] = intervals_df['relative_timestamps'].apply(
                self.bin_spikes, bin_size=self.params.binsize)
            intervals_df['num_spikes'] = intervals_df['relative_timestamps'].apply(len)
            intervals_df['duration'] = intervals_df['end_time'] - intervals_df['start_time']
            intervals_df = intervals_df[intervals_df['duration'] >= 1]
            intervals_df['mean_spike_rate'] = intervals_df['binned_spike_rates'].apply(np.mean)
            intervals_df['sem_spike_rate'] = intervals_df['binned_spike_rates'].apply(stats.sem)
        return intervals_df

    @property
    def final_stats(self):
        file_stats = self.intervals_df.groupby('event')['mean_spike_rate'].agg(['mean', 'sem']).reset_index()
        file_stats.columns = ['event', 'mean', 'sem']
        file_stats = file_stats.T
        cols = file_stats.iloc[0, :]
        file_stats = file_stats.iloc[1:, :]
        file_stats.columns = cols
        file_stats = self.reorder_stats_columns(file_stats)
        return file_stats

    @property
    def means(self):
        return self.final_stats.iloc[0, :]

    @property
    def sems(self):
        return self.final_stats.iloc[1, :]

    def reorder_stats_columns(self, df):
        existing_cols = [col for col in self.params.order if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in existing_cols]
        new_cols = existing_cols + remaining_cols
        return df[new_cols]

    def __populate(self):
        count = 0
        for var in self.nex['Variables']:
            if var['Header']['Type'] == 2:
                self.intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 0:
                self.neurons[[var['Header']['Name']][0]] = var['Timestamps']
                self.neurostamps = var['Timestamps']
                self.__neuron_df = pd.DataFrame({'timestamp': self.neurostamps})
                count += 1
        self.ensemble = True if count > 1 else False

    def get_intervals_df(self):
        intervals_list = []
        for event, (start_times, end_times) in self.intervals.items():
            if event not in ['AllFile', 'nospo']:
                for start, end in zip(start_times, end_times):
                    intervals_list.append({'event': event, 'start_time': start, 'end_time': end})
        return pd.DataFrame(intervals_list)

    @staticmethod
    def bin_spikes(relative_timestamps, bin_size=1):
        if hasattr(relative_timestamps, 'size') and relative_timestamps.size == 0:
            return []

        max_time = max(relative_timestamps)
        num_bins = int(np.ceil(max_time / bin_size))
        binned_spike_counts = np.zeros(num_bins)

        for spike_time in relative_timestamps:
            bin_index = int(spike_time // bin_size)
            binned_spike_counts[bin_index] += 1

        binned_spike_rates = binned_spike_counts / bin_size
        return binned_spike_rates.tolist()

    @staticmethod
    def get_relative_timestamps(row, neuron_df):
        start = row['start_time']
        end = row['end_time']

        mask = (neuron_df['timestamp'] >= start) & (neuron_df['timestamp'] <= end)
        event_neurons = neuron_df[mask]['timestamp'] - start
        relative_timestamps = event_neurons.to_list()
        return np.asarray(relative_timestamps)

