from __future__ import annotations
import warnings
from collections import namedtuple
import numpy as np
import scipy.stats as stats
import pandas as pd
from params import Params
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

class EatingSignals:
    def __init__(self, nexdata: dict):

        self.nex = nexdata
        self.ensemble = False
        self.params = Params()
        self.neurons: dict = {}
        self.intervals = {}
        self.__populate()

        self.event_df = self.get_event_df() # event, start_time, end_time
        self.counts = self.get_counts() # event, count (number of events)
        self.neuron_df = self.build_neuron_df() #neuron, event, mean (spike rate), sem (spike rate), count
        self.mean_df = self.neuron_df.pivot(index='neuron', columns='event', values='mean').reset_index()
        self.sem_df = self.neuron_df.pivot(index='neuron', columns='event', values='sem').reset_index()

        self.num_ts = {key: len(value) for key, value in self.neurons.items()}
        self.opto = False

    def __repr__(self) -> str:
        ens = '-E' if self.ensemble else ''
        return f'{self.__class__.__name__}{ens}'

    @staticmethod
    def get_spike_times_within_interval(timestamps, start, end):
        return timestamps[(timestamps >= start) & (timestamps <= end)]

    def get_counts(self):
        events, counts = np.unique(self.event_df['event'], return_counts=True)
        return {event: count for event, count in zip(events, counts)}

    def build_neuron_df(self):
        results = []
        # Loop through each neuron and its timestamps
        for neuron, timestamps in self.neurons.items():
            # Loop through each row in the events_df
            for _, row in self.event_df.iterrows():
                spike_times_within_interval = self.get_spike_times_within_interval(
                    timestamps, row['start_time'], row['end_time'])
                spike_rate = len(spike_times_within_interval) / (row['end_time'] - row['start_time'])
                results.append({
                    'neuron': neuron,
                    'event': row['event'],
                    'mean_spike_rate': spike_rate,
                })

        spike_rates_df = pd.DataFrame(results)

        grouped_spike_rates_df = spike_rates_df.groupby(['neuron', 'event'])
        neuron_stats = grouped_spike_rates_df.agg(
            mean=('mean_spike_rate', 'mean'),
            sem=('mean_spike_rate', stats.sem),
            count=('mean_spike_rate', 'count'),
        ).reset_index()

        return neuron_stats

    @property
    def means(self) -> pd.DataFrame:
        return self.mean_df

    @property
    def sems(self) -> pd.DataFrame:
        return self.sem_df

    def reorder_stats_columns(self, df):
        existing_cols = [col for col in self.params.order if col in df.columns]
        remaining_cols = [col for col in df.columns if col not in existing_cols]
        new_cols = existing_cols + remaining_cols
        return df[new_cols]

    def __populate(self):
        count = 0
        self.neurons = {}
        for var in self.nex['Variables']:
            if var['Header']['Type'] == 2:
                self.intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 0:
                self.neurons[var['Header']['Name']] = var['Timestamps']
                count += 1
        self.ensemble = True if count > 1 else False

    def get_event_df(self):
        intervals_list = []
        for event, (start_times, end_times) in self.intervals.items():
            if event not in ['AllFile', 'nospo', 'ALLFILE', 'start', 'Interbout_int']:
                for start, end in zip(start_times, end_times):
                    intervals_list.append({'event': event, 'start_time': start, 'end_time': end})

        intervals_df = pd.DataFrame(intervals_list)
        intervals_df = intervals_df[(intervals_df['end_time'] - intervals_df['start_time']) >= 1]
        return intervals_df

