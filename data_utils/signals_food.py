from __future__ import annotations
import os
import warnings
import numpy as np
import scipy.stats as stats
import pandas as pd
from params import Params
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

bad_events = ['eating_unknown', 'tail', 'Tail', 'e_unknown', 'Interbout_Int', 'nodata']

name_mapping = {
    'grooming': 'Grooming',
    'Grooming (1)': 'Grooming',
    'grooming (1)': 'Grooming',

    'w_apple': 'well_apple',
    'w_a': 'well_apple',
    'e_a': 'eat_apple',
    'e_apple': 'eat_apple',

    'w_nuts': 'well_nuts',
    'w_n': 'well_nuts',
    'e_nuts': 'eat_nuts',
    'e_n': 'eat_nuts',

    'w_c': 'well_choc',
    'e_c': 'eat_choc',
    'w_choc': 'well_choc',
    'e_choc': 'eat_choc',

    'w_banana': 'well_banana',
    'e_banana': 'eat_banana',

    'w_b': 'well_broccoli',
    'e_b': 'eat_broccoli',
    'w_broccoli': 'well_broccoli',
    'e_broccoli': 'eat_broccoli',
}

def get_spike_times_within_interval(timestamps, start, end):
    return timestamps[(timestamps >= start) & (timestamps <= end)]

def filter_events(df):
    df['column_name'] = df['column_name'].replace(name_mapping)

class EatingSignals:
    def __init__(self, nexdata: dict, filename: str, opto=False,):
        self.__nex = nexdata
        self.filename = filename
        self.opto = opto
        self.params = Params()
        self.neurons: dict = {}
        self.intervals = {}
        self.__populate()

        self.event_df = self.get_event_df() # event, start_time, end_time
        self.neuron_df = self.build_neuron_df()

    def __repr__(self) -> str:
        return f'{self.filename} - {self.num_neurons} neurons'

    @property
    def waveforms(self):
        return {key: len(value) for key, value in self.neurons.items()}

    @property
    def num_neurons(self):
        return len(self.neurons.keys())

    @property
    def events(self):
        events = self.event_df['event'].unique()
        return events[~np.isin(events, bad_events)]

    @property
    def means(self):
        return self.neuron_df.pivot(index='neuron', columns='event', values='mean').reset_index()

    @property
    def sems(self):
        return self.neuron_df.pivot(index='neuron', columns='event', values='sem').reset_index()

    @property
    def trial_stats(self):
        df = self.neuron_df.drop_duplicates(subset=['event', 'trials', 'mean_time', 'sem_time'])
        df = df[['event', 'trials', 'mean_time', 'sem_time']].T.reset_index()
        df['filename'] = self.filename
        df.set_index('filename', inplace=True)
        df.columns = df.iloc[0]
        df = df.iloc[1:]
        return df

    def __populate(self):
        self.neurons = {}
        for var in self.__nex['Variables']:
            if var['Header']['Type'] == 2:
                self.intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 0:
                self.neurons[var['Header']['Name']] = var['Timestamps']
        self.intervals = {
            key: value for key, value in self.intervals.items() if key not in ['allwell', 'alleat', 'allzoneend',
                                                                               'AllFile', 'nospo', 'ALLFILE',
                                                                               'start', 'Interbout_int',
                                                                               'nolaser_prewell', 'laser_prewell']
        }
        if self.opto:
            self.map_keys()

    def get_well_loc(self):
        # Use os.path.splitext to split the extension from the rest of the file name
        base_name = os.path.splitext(self.filename)[0]
        parts = base_name.split('_')
        return parts[-1]

    def map_keys(self):
        key_mapping = {
            'laserwell': 'laser_well',
            'laserzone_prewell': 'laser_prewell',
            'nolaserwell': 'nolaser_well',
            'nolaserzone_prewell': 'nolaser_prewell'
        }
        self.intervals = {k.rstrip('0_2') if k.endswith('0_2') else k: v for k, v in self.intervals.items() if
                          not k.endswith('-2')}
        self.intervals = {key_mapping.get(key, key): value for key, value in self.intervals.items()}

    def build_neuron_df(self):
        results = []
        for neuron, timestamps in self.neurons.items():
            # Loop through each row in the events_df
            for _, row in self.event_df.iterrows():
                spike_times_within_interval = get_spike_times_within_interval(
                    timestamps, row['start_time'], row['end_time'])
                spike_rate = len(spike_times_within_interval) / (row['end_time'] - row['start_time'])
                interval_duration = row['end_time'] - row['start_time']
                event_result = {
                    'neuron': neuron,
                    'event': row['event'],
                    'mean_spike_rate': spike_rate,
                    'interval_duration': interval_duration,
                    }
                results.append(event_result)
        spike_rates_df = pd.DataFrame(results)

        grouped_spike_rates_df = spike_rates_df.groupby(['neuron', 'event'])
        neuron_stats = grouped_spike_rates_df.agg(
            mean=('mean_spike_rate', 'mean'),
            sem=('mean_spike_rate', stats.sem),
            trials=('mean_spike_rate', 'count'),
            mean_time=('interval_duration', 'mean'),
            sem_time=('interval_duration', stats.sem),
        ).reset_index()
        return neuron_stats

    def get_event_df(self):
        intervals_list = []
        for event, (start_times, end_times) in self.intervals.items():
            if event not in bad_events:
                event = name_mapping.get(event, event)
                for start, end in zip(start_times, end_times):
                        intervals_list.append(
                            {'event': event, 'start_time': start, 'end_time': end})
        intervals_df = pd.DataFrame(intervals_list)
        intervals_df = intervals_df[(intervals_df['end_time'] - intervals_df['start_time']) >= 1]
        return intervals_df
