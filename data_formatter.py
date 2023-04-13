from __future__ import annotations
from typing import Optional, Tuple, Generator, List
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
        self.params = {
            "start": start,
            "end": end,
            "binsize": binsize
            }
        self.colors = colors
        self.__raw_datadict = nexdata
        self.__well_events = well_events
        self.__eating_events = eating_events

        self.neuron: str = ''
        self.neurostamps = []
        self.session_length = None
        self.__intervals = {}

        self.__well_intervals = {}
        self.__eating_intervals = {}
        self.__spont_intervals = {}
        self.df_binned = pd.DataFrame(columns=['event', 'bins', 'mid'])
        self.pt = pd.DataFrame()
        self.df_sorted = pd.DataFrame()
        self.ev_nums = pd.Series(dtype=float)
        self.trials = {}
        self.populate()
        self.sort()
        self.splice()
        self.spont_mean_std: list = self.spont_mean_std()

    def populate(self):
        for var in self.__raw_datadict['Variables']:
            if var['Header']['Type'] == 2 and var['Header']['Name'] in self.__well_events:
                self.__intervals[var['Header']['Name']] = var['Intervals']
                self.__well_intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 2 and var['Header']['Name'] in self.__eating_events:
                self.__intervals[var['Header']['Name']] = var['Intervals']
                self.__eating_intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 2 and var['Header']['Name'] in ['Spont', 'spont']:
                self.__intervals['spont'] = var['Intervals']
                self.__spont_intervals['spont'] = var['Intervals']
            if var['Header']['Type'] == 0:
                self.neuron = [var['Header']['Name']][0]
                self.neurostamps = var['Timestamps']
                self.session_length = np.round(self.neurostamps[-1] - self.neurostamps[1], 3)

    def sort(self) -> None:
        # intialize data containers
        neuro_df = pd.DataFrame(columns=['neuron', 'event', 'color', 'bins', 'counts', 'mid'])
        neuro_df['neuron'] = self.neurostamps
        neuro_df['bins'] = pd.cut(neuro_df['neuron'], int(np.round(self.session_length) * 10), include_lowest=True)
        neuro_df['counts'] = (neuro_df.groupby(['bins'])['neuron'].transform('count'))*10
        neuro_df['mid'] = neuro_df.bins.apply(lambda row: row.mid.round(decimals=3).astype(float))
        self.pt = neuro_df.pivot_table(index='neuron', values='bins', aggfunc='mean').reset_index()
        for key in self.__intervals:
            intervs = list(zip(self.__intervals[key][0], self.__intervals[key][1]))
            for interval in intervs:
                idx = np.where((neuro_df['neuron'] >= interval[0]) & (neuro_df['neuron'] <= interval[1]))[0]
                if len(idx) > 0:
                    neuro_df.iloc[idx[0]:idx[-1], 1] = key
                    neuro_df.iloc[idx[0]:idx[-1], 2] = self.colors[key]
        neuro_df.color = neuro_df.color.fillna('black')
        self.ev_nums = self.number_events(neuro_df['event'])
        self.df_sorted = neuro_df

    def number_events(self, ev_arr) -> list[int]:
        new_arr = []
        arr = np.asarray(ev_arr)
        for i, ev in enumerate(arr):
            if ev in self.__well_events:
                new_arr.append(int(1))
            elif ev in self.__eating_events:
                new_arr.append(int(2))
            elif ev in ["spont"]:
                new_arr.append(int(3))
            else:
                new_arr.append(int(0))
        return new_arr


    def splice(self):
        self.trials = {k: {} for k in self.__eating_events}
        x = 0
        for row in self.df_sorted.iterrows():
            ts_idx = row[0]
            ts_stamp = row[1]['neuron']
            ts_event = row[1]['event']

            if (ts_event in self.__well_events) & (x in [0, 2]):
                start = ts_stamp
                begin = start - 2
                x = 1

            if (ts_event in self.__well_events) & (x == 1):
                if type(self.df_sorted.loc[ts_idx + 1, 'event']) == float:
                    x = 2
            if ts_event in self.__eating_events:
                if type(self.df_sorted.loc[row[0] + 1, 'event']) == float:
                    end = ts_stamp
                    x = 0
                    trial_num = len(self.trials[ts_event].keys()) + 1
                    idx = np.where((self.df_sorted['neuron'] >= begin) & (self.df_sorted['neuron'] <= end))[0]
                    self.trials[ts_event][trial_num] = self.df_sorted.loc[idx[0]:idx[-1], :]

    def spont_mean_std(self) -> list:
        avg, std = [], []
        intervs = list(zip(self.__spont_intervals["spont"][0], self.__spont_intervals["spont"][1]))
        for interval in intervs:
            hist = np.histogram(
                self.neurostamps,
                bins=np.arange(
                    interval[0],
                    interval[1], self.params["binsize"]))[0]
            avg.append(statistics.mean(hist)*10)
            std.append(statistics.stdev(hist.astype(float)))

        return [statistics.mean(avg), statistics.stdev(std)]

    def generate_intervals(self, key: str):
        """Returns a generator with each interval for a given key."""
        intervs = list(zip(self.__spont_intervals[key][0], self.__spont_intervals[key][1]))
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

    def generate_spectrogram_data(self):
        for ev in self.trials.keys():
            x_bin =  np.arange(-2, 5, 0.1)
            if self.trials[ev]:
                spect = np.zeros(shape=(len(self.trials[ev]), len(x_bin)-1))
                for trial, this_trial_df in self.trials[ev].items():
                    trial_length = (this_trial_df["neuron"].iloc[-1] - this_trial_df["neuron"].iloc[0])
                #     bins = pd.cut(this_trial_df.iloc[:,0], int(np.round(trial_length)*10))
                #     counts = this_trial_df.groupby(['bins'])['color'].transform('count')
                #     this_trial_df.loc[:, "bins"] = bins
                #     self.trials[ev][trial] = this_trial_df
                #     # self.trials[ev][trial].loc[:,"Bin"] = pd.cut(this_trial_df.iloc[:,0], int(np.round(trial_length)*10))
                #     hist, bins = np.histogram(trial_spikes, x_bin)
                #     spect[trial-1,:] = hist*10
                #
                # yield spect, this_trial_df.iloc[:,3], trial_length
