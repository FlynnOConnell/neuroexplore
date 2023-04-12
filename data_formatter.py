from __future__ import annotations
from typing import Optional, Tuple, Generator
from collections import namedtuple
from nex import nexfile
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import statistics
from helpers import funcs, ax_helpers
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.DEBUG, format="%(message)s")
logger.setLevel(logging.DEBUG)


class DataFormatter:
    """
    File handler. From directory, return genorator looping through files.
    Parameters:
    ___________
    """
    def __init__(
            self,
            nexdata: dict,
            well_events,
            eating_events,
            colors,
            start: int = -1,
            end: int = 6,
            binsize: int | float = 0.1

    ) -> None:
        self.colors = colors
        self.raw_datadict = nexdata
        self.well_events = well_events
        self.eating_events = eating_events
        self.end_trial = end
        self.start_trial = start
        self.binsize = binsize
        self.neuron: str = ''
        self.neurostamps = []
        self.intervals = {}

        self.well_intervals = {}
        self.eating_intervals = {}
        self.spont_intervals = {}
        self.df_sorted = {}
        self.trials = {}
        self.spont = None
        self.populate()
        self.sort()
        self.splice()
        self.spont_mean_std: list = self.spont_mean_std()

    def populate(self):
        for var in self.raw_datadict['Variables']:
            if var['Header']['Type'] == 2 and var['Header']['Name'] in self.well_events:
                self.intervals[var['Header']['Name']] = var['Intervals']
                self.well_intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 2 and var['Header']['Name'] in self.eating_events:
                self.intervals[var['Header']['Name']] = var['Intervals']
                self.eating_intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 2 and var['Header']['Name'] in ['Spont']:
                self.intervals[var['Header']['Name']] = var['Intervals']
                self.spont_intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 0:
                self.neuron = [var['Header']['Name']][0]
                self.neurostamps = var['Timestamps']

    def sort(self) -> None:
        # intialize data containers
        neuro_df = pd.DataFrame(columns=['Neuron', 'Event', 'Color'])
        neuro_df["Neuron"] = self.neurostamps
        for key in self.intervals:
            intervs = list(zip(self.intervals[key][0], self.intervals[key][1]))
            for interval in intervs:

                # hist = np.histogram(neur, bins=np.arange(interval[0] + start, interval[1], binsize))[0]
                idx = np.where((neuro_df['Neuron'] >= interval[0]) & (neuro_df['Neuron'] <= interval[1]))[0]
                if len(idx) > 0:
                    neuro_df.iloc[idx[0]:idx[-1], 1] = key
                    neuro_df.iloc[idx[0]:idx[-1], 2] = self.colors[key]
        neuro_df.Color = neuro_df.Color.fillna('black')
        self.df_sorted[self.neuron] = neuro_df

    def splice(self):
        for neuron, neuro_df in self.df_sorted.items():
            self.trials = {k: {} for k in self.eating_events}
            x = 0
            for row in neuro_df.iterrows():
                ts_idx = row[0]
                ts_stamp = row[1]['Neuron']
                ts_event = row[1]['Event']

                if (ts_event in self.well_events) & (x in [0, 2]):
                    start = ts_stamp
                    begin = start - 2
                    x = 1

                if (ts_event in self.well_events) & (x == 1):
                    if type(neuro_df.loc[ts_idx + 1, 'Event']) == float:
                        x = 2
                if ts_event in self.eating_events:
                    if type(neuro_df.loc[row[0] + 1, 'Event']) == float:
                        end = ts_stamp
                        x = 0
                        trial_num = len(self.trials[ts_event].keys()) + 1
                        idx = np.where((neuro_df['Neuron'] >= begin) & (neuro_df['Neuron'] <= end))[0]
                        self.trials[ts_event][trial_num] = neuro_df.loc[idx[0]:idx[-1], :]

    def spont_mean_std(self) -> list:
        avg, std = [], []
        intervs = list(zip(self.spont_intervals["Spont"][0], self.spont_intervals["Spont"][1]))
        for interval in intervs:
            hist = np.histogram(
                self.neurostamps,
                bins=np.arange(
                    interval[0],
                    interval[1], self.binsize))[0]
            avg.append(statistics.mean(hist)*10)
            std.append(statistics.stdev(hist.astype(float)))

        return [statistics.mean(avg), statistics.stdev(std)]

    def generate_intervals(self, key: str):
        """Returns a generator with each interval for a given key."""
        intervs = list(zip(self.spont_intervals[key][0], self.spont_intervals[key][1]))
        for interv in intervs:
            yield interv

    def generate_trials_strict(self) -> Generator[Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Returns a generator with a dataframe for each trial, previous trial is cut-off.
        Only nan values for pre-stim timeframe.
        """
        for event, trial in self.trials.items():
            for trial_num, df in trial.items():
                ts = df.iloc[:, 0]
                ev = df.iloc[:, 1]
                co = df.iloc[:, 2]
                yield ts, ev, co