from __future__ import annotations
import os
import warnings
import numpy as np
import scipy.stats as stats
import pandas as pd
from params import Params
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

name_mapping = {
    'grooming': 'Grooming',
    'Grooming (1)': 'Grooming',

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
        self.nex = nexdata
        self.filename = filename
        self.opto = opto
        self.ensemble = False
        self.params = Params()
        self.neurons: dict = {}
        self.intervals = {}
        self.__populate()
        self.num_ts = {key: len(value) for key, value in self.neurons.items()}

        self.event_df = self.get_event_df() # event, start_time, end_time
        self.counts = self.get_counts() # event, count (number of events)
        self.neuron_df = self.build_neuron_df()
        self.all_events = np.unique(self.neuron_df.event)
        self.mean_df = self.neuron_df.pivot(index='neuron', columns='event', values='mean').reset_index()
        self.sem_df = self.neuron_df.pivot(index='neuron', columns='event', values='sem').reset_index()
        self.trials_df = self.neuron_df.pivot(index='neuron', columns='event', values='trials').reset_index()
        self.mean_time_df = self.neuron_df.pivot(index='neuron', columns='event', values='mean_time').reset_index()
        self.mean_sem_df = self.neuron_df.pivot(index='neuron', columns='event', values='sem_time').reset_index()

    def __repr__(self) -> str:
        ens = '-E' if self.ensemble else ''
        return f'{self.__class__.__name__}{ens}'

    def get_counts(self):
        events, counts = np.unique(self.event_df['event'], return_counts=True)
        return {event: count for event, count in zip(events, counts)}

    def __populate(self):
        neuron_count = 0
        self.neurons = {}
        for var in self.nex['Variables']:
            if var['Header']['Type'] == 2:
                self.intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 0:
                self.neurons[var['Header']['Name']] = var['Timestamps']
                neuron_count += 1
        self.ensemble = True if neuron_count > 1 else False
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
            for start, end in zip(start_times, end_times):
                    intervals_list.append(
                        {'event': event, 'start_time': start, 'end_time': end})
        intervals_df = pd.DataFrame(intervals_list)
        intervals_df = intervals_df[(intervals_df['end_time'] - intervals_df['start_time']) >= 1]
        return intervals_df

