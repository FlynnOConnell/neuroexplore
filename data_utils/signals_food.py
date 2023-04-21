from __future__ import annotations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import Tuple, Generator
from pathlib import Path
import numpy as np
import scipy.stats as stats
import pandas as pd
from params import Params

class Signals:
    def __init__(
            self,
            __nexdata: dict,
    ) -> None:

        self.__nexdata = __nexdata
        self.params = Params()
        self.neuron: str = ''
        self.neurostamps = []
        self.intervals = {}
        self.__populate()

        self.intervals_df = pd.DataFrame(columns=['event', 'start_time', 'end_time'])
        self.get_intervals_df()
        self.intervals_df['relative_timestamps'] = self.intervals_df.apply(self.get_relative_timestamps, axis=1, neuron_df=self.__neuron_df)
        self.intervals_df['binned_spike_rates'] = self.intervals_df['relative_timestamps'].apply(self.bin_spikes,
                                                                                                 bin_size=self.params.binsize)
        self.intervals_df['num_spikes'] = self.intervals_df['relative_timestamps'].apply(len)
        self.intervals_df['duration'] = self.intervals_df['end_time'] - self.intervals_df['start_time']
        self.intervals_df = self.intervals_df[self.intervals_df['duration'] >= 1]

        self.intervals_df['mean_spike_rate'] = self.intervals_df['binned_spike_rates'].apply(np.mean)
        self.intervals_df['sem_spike_rate'] = self.intervals_df['binned_spike_rates'].apply(stats.sem)
        self.final_stats = pd.DataFrame(columns=['Spont', 'mean', 'sem'])
        self.file_stats = self.intervals_df.groupby('event')['mean_spike_rate'].agg(['mean', 'sem']).reset_index()

        self.file_stats.columns = ['event', 'mean', 'sem']
        self.file_stats = self.file_stats.T
        cols =  self.file_stats.iloc[0, :]
        self.file_stats =  self.file_stats.iloc[1:, :]
        self.file_stats.columns = cols
        self.file_stats = self.reorder_stats_columns(self.file_stats)
        self.means = self.file_stats.iloc[0, :]
        self.sems =  self.file_stats.iloc[1, :]


    def reorder_stats_columns(self, df):
        existing_cols = [col for col in self.params.order if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in existing_cols]
        new_cols = existing_cols + remaining_cols
        return df[new_cols]

    def __populate(self):
        for var in self.__nexdata['Variables']:
            if var['Header']['Type'] == 2:
                self.intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 0:
                self.neuron = [var['Header']['Name']][0]
                self.neurostamps = var['Timestamps']
                self.__neuron_df = pd.DataFrame({'timestamp': self.neurostamps})

    def get_intervals_df(self):
        intervals_list = []
        for event, (start_times, end_times) in self.intervals.items():
            if event not in ['AllFile', 'nospo']:
                for start, end in zip(start_times, end_times):
                    intervals_list.append({'event': event, 'start_time': start, 'end_time': end})
        self.intervals_df = pd.DataFrame(intervals_list)

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

