from __future__ import annotations
import os
import warnings
import numpy as np
import scipy.stats as stats
import pandas as pd
from params import Params
from functools import partial
import fnmatch
from data_utils.parser import PLACE_NAME_MAPPING, INCORRECT_NAME_MAPPING
from pathlib import Path
from data_utils.file_handling import get_nex
from helpers.heatmap import sf_heatmap

warnings.simplefilter(action='ignore', category=RuntimeWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)

bad_events = ['*unknown*', 'tail*', '*Tail*', '*Interbout*',
              '*data*', 'es*', '*cheerios*', '*banana*', '*mush*',
              '*cheese*', 'e_cherrios', 'eat_empty', 'eat_e']


def is_bad_event(event):
    for pattern in bad_events:
        if fnmatch.fnmatch(event, pattern):
            return True
    return False


def get_spike_times_within_interval(timestamps, start, end):
    return timestamps[(timestamps >= start) & (timestamps <= end)]


class Signals:
    def __init__(self, nexdata: dict, filename: str, ):
        self.__nex = nexdata
        self.filename = filename
        self.params = Params()
        self.neurons: dict = {}
        self.intervals = {}
        self.__populate()

        self.event_df = self.get_event_df()
        self.stats_df = self.build_stats_df()
        self.test_df = self.get_test_df()

    def get_test_df(self):
        df = self.event_df.copy()
        ev_dict = {}

        for neuron in self.neurons.keys():
            result_df = pd.DataFrame(
                columns=['event', 'well_ts', 'eat_ts'])

            for i in range(len(df) - 1):
                current_row = df.iloc[i]
                next_row = df.iloc[i + 1]

                if current_row['event'].startswith('well_') and next_row['event'].startswith('eat_'):
                    current_event = current_row['event'].split('_')[1]
                    next_event = next_row['event'].split('_')[1]

                    if current_event == next_event:
                        well_ts = get_spike_times_within_interval(
                            self.neurons[neuron], current_row['start_time'], current_row['end_time'])
                        eat_ts = get_spike_times_within_interval(
                            self.neurons[neuron], next_row['start_time'], next_row['end_time'])

                        result_df = result_df.append({
                            'event': current_event,
                            'well_ts': well_ts,
                            'eat_ts': eat_ts
                        }, ignore_index=True)

            ev_dict[neuron] = result_df
        return ev_dict

    def __repr__(self) -> str:
        return f'{self.filename} - {self.num_neurons} neurons'

    @property
    def num_neurons(self):
        return len(self.neurons.keys())

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

    def get_well_loc(self):
        # Use os.path.splitext to split the extension from the rest of the file name
        base_name = os.path.splitext(self.filename)[0]
        parts = base_name.split('_')
        return parts[-1]

    def get_spontaneous_times(self, sort, inner_gap, pad=10):
        return self.get_spont_intervs(sort, inner_gap, pad)

    def build_stats_df(self):
        results = []
        for neuron, timestamps in self.neurons.items():
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

        std_partial = partial(np.std, ddof=1)
        grouped_spike_rates_df = spike_rates_df.groupby(['neuron', 'event'])
        neuron_stats = grouped_spike_rates_df.agg(
            mean=('mean_spike_rate', 'mean'),
            sem=('mean_spike_rate', stats.sem),
            std=('mean_spike_rate', std_partial),
            trials=('mean_spike_rate', 'count'),
            mean_time=('interval_duration', 'mean'),
            sem_time=('interval_duration', stats.sem),
        ).reset_index()
        return neuron_stats

    @staticmethod
    def transform_event(event, row_name_map):
        """
        Transforms event name to match the row name in the interval dataframe
        """
        prefix = event[:1]  # first letter of event: e will be eating, F,and B will both be well locations
        mapper = None
        new_prefix = None
        if prefix == 'e':
            new_prefix = "eat_"
            mapper = event[-2:]  # last 2 letters of event: Int, Out, In
        elif prefix == 'F' or prefix == 'B':
            new_prefix = "well_"
            mapper = event[:2]  # first 2 letters of event: FL, FR, BL, BR
        else:
            return event  # if prefix is not e, F, or B, return original event
        # Checking prefix in row_name_map and replacing it, if not found keeping original
        new_ev = row_name_map.get(mapper, prefix)
        new_event = new_prefix + new_ev
        return new_event

    def parse_dataframe(self, input_df):

        # Check if the filename is in the mapping of files with food well locations.
        if self.filename in PLACE_NAME_MAPPING:
            # Get the row name mapping for this filename.
            row_name_map = PLACE_NAME_MAPPING[self.filename]

            # Create a new event column based on the current one
            input_df['new_event'] = input_df['event'].apply(
                lambda event: self.transform_event(event, row_name_map))
            input_df = input_df.drop(columns=['event'])
            input_df = input_df.rename(columns={'new_event': 'event'})
        return input_df

    def get_event_df(self):
        intervals_list = []
        for event, (start_times, end_times) in self.intervals.items():
            for start, end in zip(start_times, end_times):
                intervals_list.append(
                    {'event': event, 'start_time': start, 'end_time': end})
        intervals_df = pd.DataFrame(intervals_list)
        intervals_df = self.parse_dataframe(intervals_df)
        intervals_df = intervals_df[~intervals_df['event'].apply(is_bad_event)]
        intervals_df['event'] = intervals_df['event'].map(INCORRECT_NAME_MAPPING).fillna(intervals_df['event'])
        intervals_df.sort_values(by='start_time', inplace=True)
        return intervals_df

    def get_spont_intervs(self, sort=True, spont_gap=16, pre_event_gap=3, ts=False):
        if self.event_df is None:
            raise ValueError("Event dataframe is not initialized. Call get_event_df() first.")

        df = self.event_df
        spont_list = []

        df = df.sort_values(by='start_time')

        # Loop through dataframe rows
        for i in range(1, df.shape[0]):
            prev_event = df.iloc[i - 1]
            curr_event = df.iloc[i]

            # Check if there's enough gap for a spontaneous event
            if curr_event['start_time'] - prev_event['end_time'] >= spont_gap:
                # Add spontaneous event starting 3 seconds after the previous event and ending just before the
                # current event
                spont_list.append({'event': 'spont',
                                   'start_time': prev_event['end_time'] + 3,
                                   'end_time': curr_event['start_time'] - 3
                                   })

        # Create a dataframe from the list
        return pd.DataFrame(spont_list)

    def get_spont_stamps(self, spont_intervs):
        # Create an empty DataFrame to store results
        result_df = pd.DataFrame(columns=['neuron', 'mean_spike_rate', 'std_spike_rate', 'sem_spike_rate'])

        for neuron in self.neurons:
            timestamps = self.neurons[neuron]
            spont_intervs = self.get_spont_intervs(self.event_df)
            spont_spike_rates = []
            bin_width = 1.0  # Width of bins in seconds
            for _, row in spont_intervs.iterrows():
                spike_times_within_interval = get_spike_times_within_interval(
                    timestamps, row['start_time'], row['end_time'])
                num_bins = int((row['end_time'] - row['start_time']) / bin_width)
                if (row['end_time'] - row['start_time']) % bin_width >= bin_width / 2:
                    num_bins += 1  # Round up if remaining interval is at least half the bin width
                spikes_per_bin, _ = np.histogram(spike_times_within_interval, bins=num_bins)
                spike_rates_per_bin = spikes_per_bin / bin_width  # Convert to rates
                spont_spike_rates.extend(spike_rates_per_bin)

            # Calculate mean, standard deviation and standard error of the mean of spontaneous spike rates for
            # current neuron
            mean_spike_rate = np.mean(spont_spike_rates)
            std_spike_rate = np.std(spont_spike_rates)
            sem_spike_rate = std_spike_rate / np.sqrt(len(spont_spike_rates))

            new_row = pd.DataFrame({'neuron': [neuron],
                                    'mean_spike_rate': [mean_spike_rate],
                                    'std_spike_rate': [std_spike_rate],
                                    'sem_spike_rate': [sem_spike_rate]})
            # Add the results to the dataframe
            result_df = pd.concat([result_df, new_row], ignore_index=True)
        return result_df


if __name__ == "__main__":
    data_dir = Path.home() / 'data' / 'sf' / 'SFN07_2018-05-04_SF.nex'
    nex_file = get_nex(data_dir)
    data = Signals(nex_file, data_dir.stem)
    test_df = data.test_df
    for neuron in test_df:
        sf_heatmap(test_df[neuron])

    # Plotting two side-by-side heatmaps
    # fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    # left heatmap is
