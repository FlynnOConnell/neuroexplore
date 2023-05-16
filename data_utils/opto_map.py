from __future__ import annotations
import warnings
import numpy as np
import scipy.stats as stats
import pandas as pd
from params import Params
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

def get_spike_times_within_interval(timestamps, start, end):
    return timestamps[(timestamps >= start) & (timestamps <= end)]

class EatingSignals:
    def __init__(self, nexdata: dict, opto=False):
        self.nex = nexdata
        self.opto = opto
        self.ensemble = False
        self.params = Params()
        self.neurons: dict = {}
        self.intervals = {}
        self.__populate()
        self.num_ts = {key: len(value) for key, value in self.neurons.items()}

        self.event_df = self.get_event_df() # event, start_time, end_time
        self.counts = self.get_counts() # event, count (number of events)
        self.neuron_df, self.laser_df  = self.build_neuron_df() #neuron, event, mean (spike rate), sem (spike rate), count
        self.mean_df = self.neuron_df.pivot(index='neuron', columns='event', values='mean').reset_index()
        self.sem_df = self.neuron_df.pivot(index='neuron', columns='event', values='sem').reset_index()

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
        self.intervals = {key: value for key, value in self.intervals.items() if key not in ['allwell', 'alleat', 'allzoneend', 'AllFile', 'nospo', 'ALLFILE', 'start', 'Interbout_int']}

        if self.opto:
            self.map_keys()

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

    def get_opto_start_end_times(self, laser=True):
        if not self.opto:
            raise ValueError('Opto data not configured as true.')
        if laser:
            keys = ['laser_eat', 'laser_well', 'laser_prewell']
        else:
            keys = ['nolaser_eat', 'nolaser_well', 'nolaser_prewell']

        # Create a nested list where each ndarray is converted into a list
        nested_list = [list(self.intervals[key][0]) for key in keys if key in self.intervals]
        return [item for sublist in nested_list for item in sublist]

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
                    'type': row['event_type'],
                    }
                if self.opto:
                    event_result['laser'] = row['laser']
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

        grouped_spike_rates_df_laser = spike_rates_df.groupby(['neuron', 'event', 'laser'])
        neuron_stats_laser = grouped_spike_rates_df_laser.agg(
            mean=('mean_spike_rate', 'mean'),
            sem=('mean_spike_rate', stats.sem),
            trials=('mean_spike_rate', 'count'),
            mean_time=('interval_duration', 'mean'),
            sem_time=('interval_duration', stats.sem),
        ).reset_index()

        return neuron_stats, neuron_stats_laser

    def get_event_df(self):
        intervals_list = []
        laser_times_true = self.get_opto_start_end_times(laser=True) if self.opto else []
        laser_times_false = self.get_opto_start_end_times(laser=False) if self.opto else []

        for event, (start_times, end_times) in self.intervals.items():
            for start, end in zip(start_times, end_times):
                # Check the event type
                if event in self.params.well_events:
                    event_type = 'well'
                elif event in self.params.eating_events:
                    event_type = 'eating'
                elif event in ['Spont', 'spont']:
                    event_type = 'spont'
                else:
                    continue

                if self.opto:
                    if event_type in ['Spont', 'spont']:
                        continue
                    elif start in laser_times_true:
                        laser = True
                    elif start in laser_times_false:
                        laser = False
                    else:
                        laser = 'zone'
                    intervals_list.append(
                        {'event': event, 'start_time': start, 'end_time': end, 'event_type': event_type, 'laser': laser})
                else:
                    intervals_list.append(
                        {'event': event, 'start_time': start, 'end_time': end, 'type': event_type})

        intervals_df = pd.DataFrame(intervals_list)
        intervals_df = intervals_df[(intervals_df['end_time'] - intervals_df['start_time']) >= 1]

        return intervals_df

