import os
import numpy as np
from .data_formatter import DataFormatter
from params import Params
from nex.nexfile import Reader

class Data:
    def __init__(self, nexfile):
        self.params = Params()
        self.__format(nexfile)

    def __repr__(self):
        return f'Neuron: {self.__alldata.neuron})'

    def __format(self, nexfile):
        self.__alldata = DataFormatter(nexfile, self.params.well_events, self.params.eating_events, self.params.colors)
        self.__get_data()

    def __get_data(self):
        self.__alldata = self.__alldata
        self.spikes = self.__alldata.neurostamps
        self.trials = self.__alldata.trials
        self.neuron = self.__alldata.neuron
        self.sorted = self.__alldata.df_sorted
        self.binned = self.__alldata.df_binned
        self.means = { event: self.get_mean_sps(event) for event in self.params.well_events + self.params.eating_events }
        self.stds = { event: self.get_std_sps(event) for event in self.params.well_events + self.params.eating_events }

    def get_mean_sps(self, event):
        filtered_df = self.binned[self.binned['event'] == event]
        if len(filtered_df) < 2:
            return "Too few values to calulate mean"
        total_spike_count = filtered_df['count'].sum()
        total_duration_seconds = self.sorted['neuron'].max() - self.sorted['neuron'].min()
        return total_spike_count / total_duration_seconds

    def get_std_sps(self, event):
        filtered_df = self.binned[self.binned['event'] == event]
        if len(filtered_df) < 2:
            return "Too few values to calulate std"
        spikes_per_bin = filtered_df['count'].values
        return np.std(spikes_per_bin / 0.1)

