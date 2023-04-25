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
    def __init__(self, nexdata: dict) -> None:
        self.nex = nexdata
        self.ensemble = False
        self.params = Params()
        self.neurons: dict = {}
        self.intervals = {}
        self.__populate()
        self.event_df = self.get_event_df()
        self.neuron_df = self.build_neuron_df()
        self.mean_df = pd.DataFrame()
        self.sem_df = pd.DataFrame()
        self.update_mean_and_sem_dataframes(self.neuron_df)
        self.num_ts = {key: len(value) for key, value in self.neurons.items()}

    def __repr__(self) -> str:
        ens = '-E' if self.ensemble else ''
        return f'{self.__class__.__name__}{ens}'

    @staticmethod
    def get_spike_times_within_interval(timestamps, start, end):
        return timestamps[(timestamps >= start) & (timestamps <= end)]

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
                    'mean_spike_rate': spike_rate
                })

        # Convert the list to a DataFrame
        spike_rates_df = pd.DataFrame(results)

        # Create a new DataFrame to store the results for each neuron and event
        final_results_df = pd.DataFrame(columns=['neuron', 'event', 'mean', 'sem'])

        for neuron in spike_rates_df['neuron'].unique():
            for event in spike_rates_df['event'].unique():
                mask = (spike_rates_df['neuron'] == neuron) & (spike_rates_df['event'] == event)
                mean_spike_rate = spike_rates_df.loc[mask, 'mean_spike_rate'].mean()
                sem_spike_rate = stats.sem(spike_rates_df.loc[mask, 'mean_spike_rate'])

                final_results_df = final_results_df.append({
                    'neuron': neuron,
                    'event': event,
                    'mean': mean_spike_rate,
                    'sem': sem_spike_rate
                }, ignore_index=True)

        return final_results_df

    def update_mean_and_sem_dataframes(self, final_results_df):
        unique_neurons = np.unique(final_results_df['neuron'])
        unique_events = np.unique(final_results_df['event'])

        if self.mean_df.empty:
            self.mean_df = pd.DataFrame({'neuron': unique_neurons})
            self.sem_df = pd.DataFrame({'neuron': unique_neurons})

        for event in unique_events:
            mean_values = []
            sem_values = []

            for neuron in unique_neurons:
                mask = (final_results_df['neuron'] == neuron) & (final_results_df['event'] == event)
                mean_value = final_results_df.loc[mask, 'mean'].values[0]
                sem_value = final_results_df.loc[mask, 'sem'].values[0]

                mean_values.append(mean_value)
                sem_values.append(sem_value)

            self.mean_df[event] = mean_values
            self.sem_df[event] = sem_values

    @property
    def means(self):
        return self.mean_df

    @property
    def sems(self):
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
