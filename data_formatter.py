from __future__ import annotations

from nex import nexfile

from helpers import funcs, ax_helpers
import logging
from collections import namedtuple
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

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
            colors

    ) -> None:
        self.colors = colors
        self.raw_datadict = nexdata
        self.well_events = well_events
        self.eating_events = eating_events
        self.neurons = ()
        self.intervals = {}
        self.timestamps = {}
        self.well_intervals = {}
        self.eating_intervals = {}
        self.spont_intervals = {}
        self.neuroframes = {}
        self.trials = {}
        self.spont = None
        self.populate()
        self.sort()
        self.splice()
        self.spont_binned = self.spont_bins()


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
                self.neurons = self.neurons + tuple([var['Header']['Name']])
                self.timestamps[var['Header']['Name']] = var['Timestamps']

    def sort(self) -> None:
        for neuron in self.neurons:
            # intialize data containers
            neur = self.timestamps[neuron]
            neuro_df = pd.DataFrame(columns=['Neuron', 'Event', 'Color'])
            neuro_df["Neuron"] = neur
            for key in self.intervals:
                intervs = list(zip(self.intervals[key][0], self.intervals[key][1]))
                for interval in intervs:
                    # hist = np.histogram(neur, bins=np.arange(interval[0] + start, interval[1], binsize))[0]
                    idx = np.where((neuro_df['Neuron'] >= interval[0]) & (neuro_df['Neuron'] <= interval[1]))[0]
                    neuro_df.loc[idx[0]:idx[-1], 'Event'] = key
                    neuro_df.loc[idx[0]:idx[-1], 'Color'] = self.colors[key]
            neuro_df.Color = neuro_df.Color.fillna('black')
            self.neuroframes[neuron] = neuro_df

    def splice(self):
        for neuron, neuro_df in self.neuroframes.items():
            self.trials = {k: {} for k in self.eating_events}
            counter = 0
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
                        counter += 1
                        idx = np.where((neuro_df['Neuron'] >= begin) & (neuro_df['Neuron'] <= end))[0]
                        self.trials[ts_event][counter] = neuro_df.loc[idx[0]:idx[-1], :]

    def spont_bins(self):
        spont_hist = []
        intervs = list(zip(self.spont_intervals["Spont"][0], self.spont_intervals["Spont"][1]))
        for interval in intervs:
            start = -1
            binsize = 0.1
            hist = np.histogram(
                self.timestamps['SPK08a'],
                bins=np.arange(
                    interval[0],
                    interval[1], binsize))[0]
            mean = zip(hist, )
            spont_hist.append(hist)
        return spont_hist
