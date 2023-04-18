from __future__ import annotations
import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
from typing import Tuple, Generator
import numpy as np
import pandas as pd

class DataFormatter:
    """
    DataFormatter. Data manipulation and handling.

    Parameters:
    -----------
    nexdata : dict
        Dictionary containing the raw data in NEX format.
    well_events : list
        List of well event names.
    eating_events : list
        List of eating event names.
    colors : dict
        Dictionary mapping event names to colors.
    start : int, optional, default: -1
        Start time for data formatting in seconds. Defaults to -1.
    end : int, optional, default: 6
        End time for data formatting in seconds. Defaults to 6.
    binsize : int or float, optional, default: 0.1
        Binsize for data formatting in seconds. Defaults to 0.1.

    Attributes:
    -----------
    params : dict
        Dictionary containing the parameters used for data formatting.
    colors : dict
        Dictionary mapping event names to colors.
    raw : dict
        Dictionary containing the raw data in NEX format.
    neuron : str
        Name of the neuron.
    neurostamps : list
        List of timestamps for the neuron.
    session_length : float
        Length of the session in seconds.
    pt : pd.DataFrame
        DataFrame containing pivot table data.
    df_sorted : pd.DataFrame
        DataFrame containing sorted data.
    trials : dict
        Dictionary containing each [-2s, well, eating] trial.

    Methods:
    --------
    binned_spont_mean_std()
        Calculates the mean and standard deviation of spontaneous events.
    plot_traces(neuron, start, end, ax)
        Plots the traces for a given neuron.
    plot_trial(trial_num, event_type, ax)
        Plots a trial for a given trial number and event type.
    plot_histogram(event_type, bins, ax)
        Plots a histogram for a given event type and bins.
    plot_psth(event_type, bins, ax)
        Plots a peri-stimulus time histogram (PSTH) for a given event type and bins.

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
            "binsize": binsize,
            "well_events": well_events,
            "eating_events": eating_events,
            }
        self.colors = colors
        self.raw = nexdata

        self.neuron: str = ''
        self.neurostamps = []
        self.session_length = None
        self.__intervals = {}

        self.__well_intervals = {}
        self.__eating_intervals = {}
        self.__spont_intervals = {}
        self.pt = pd.DataFrame()
        self.df_sorted = pd.DataFrame()
        self.trials = {}
        self.__populate()
        self.__sort()
        self.__validate_RF()
        self.__splice()
        self.df_binned = self.bin()
        self.add_trial_column()

    def __populate(self):
        for var in self.raw['Variables']:
            if var['Header']['Type'] == 2 and var['Header']['Name'] in self.params['well_events']:
                self.__intervals[var['Header']['Name']] = var['Intervals']
                self.__well_intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 2 and var['Header']['Name'] in self.params['eating_events']:
                self.__intervals[var['Header']['Name']] = var['Intervals']
                self.__eating_intervals[var['Header']['Name']] = var['Intervals']
            if var['Header']['Type'] == 2 and var['Header']['Name'] in ['Spont', 'spont']:
                self.__intervals['spont'] = var['Intervals']
                self.__spont_intervals['spont'] = var['Intervals']
            if var['Header']['Type'] == 0:
                self.neuron = [var['Header']['Name']][0]
                self.neurostamps = var['Timestamps']
                self.session_length = np.round(self.neurostamps[-1] - self.neurostamps[1], 3)

    def __validate_RF(self):
        if self.df_sorted['event'].isnull().values.all():
            raise ValueError('No events, likely a repeated stim file.')

    def __sort(self) -> None:
        # intialize data containers
        neuro_df = pd.DataFrame(columns=['neuron', 'event', 'color'])
        neuro_df['neuron'] = self.neurostamps
        for key in self.__intervals:
            intervs = list(zip(self.__intervals[key][0], self.__intervals[key][1]))
            for interval in intervs:
                idx = np.where((neuro_df['neuron'] >= interval[0]) & (neuro_df['neuron'] <= interval[1]))[0]
                if len(idx) > 0:
                    neuro_df.iloc[idx[0]:idx[-1], 1] = key
                    neuro_df.iloc[idx[0]:idx[-1], 2] = self.colors[key]
        neuro_df.color = neuro_df.color.fillna('black')
        self.df_sorted = neuro_df

    def add_trial_column(self) -> None:
        self.df_sorted['trial'] = (self.df_sorted['event'] != self.df_sorted['event'].shift()).groupby(self.df_sorted['event']).cumsum()
        return None

    def __splice(self):
        self.trials = {k: {} for k in self.params['eating_events']}
        x = 0
        for row in self.df_sorted.iterrows():
            ts_idx = row[0]
            ts_stamp = row[1]['neuron']
            ts_event = row[1]['event']

            if (ts_event in self.params['well_events']) & (x in [0, 2]):
                start = ts_stamp
                begin = start - 2
                x = 1

            if (ts_event in self.params['well_events']) & (x == 1):
                if type(self.df_sorted.loc[ts_idx + 1, 'event']) == float:
                    x = 2
            if ts_event in self.params['eating_events']:
                if type(self.df_sorted.loc[row[0] + 1, 'event']) == float:
                    end = ts_stamp
                    x = 0
                    trial_num = len(self.trials[ts_event].keys()) + 1
                    idx = np.where((self.df_sorted['neuron'] >= begin) & (self.df_sorted['neuron'] <= end))[0]
                    self.trials[ts_event][trial_num] = self.df_sorted.loc[idx[0]:idx[-1], :]

    def bin(self):
        bin_edges = [i * 0.1 for i in range(int(self.df_sorted['neuron'].max() // 0.1) + 2)]
        # Assign each spike time to a bin
        self.df_sorted['bin'] = pd.cut(self.df_sorted['neuron'], bins=bin_edges, include_lowest=True)
        # Group by bin and event, and count the number of spikes
        binned_data = self.df_sorted.groupby(['bin', 'event']).size().reset_index(name='count')
        return binned_data.loc[binned_data.groupby('bin')['count'].idxmax()].reset_index(drop=True)

    def generate_intervals(self, key: str):
        """
        Generate intervals for a given key.

        Returns a generator that yields each interval for the given key.

        Args:
            key (str): The key for which intervals are to be generated.

        Yields:
            Tuple[float, float]: A tuple containing two float values representing the start and end points of each interval.

        Example:
            >>> for interv in generate_intervals('key'):
            ...     # Do something with interv
        """
        intervs = list(zip(self.__spont_intervals[key][0], self.__spont_intervals[key][1]))
        for interv in intervs:
            yield interv

    def generate_trials_strict(self) -> Generator[Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        Generate trials with strict criteria.

        Returns a generator that yields three pandas Series objects for each trial, where the previous trial is
        cut-off and only NaN values are present for the pre-stimulus timeframe.

        Yields:
        Tuple[pd.Series, pd.Series, pd.Series]: A tuple containing three pandas Series objects representing the
        generated trials.

        Example:
            >>> for ts, ev, co in generate_trials_strict():
            ...     # Do something with ts, ev, co
        """
        for event, trial in self.trials.items():
            for trial_num, df in trial.items():
                ts = df.iloc[:, 0]
                ev = df.iloc[:, 1]
                co = df.iloc[:, 2]
                yield ts, ev, co

    def generate_trials_loose(self) -> Generator[Tuple[pd.Series, pd.Series, pd.Series]]:
        """
        """
        for event, trial in self.trials.items():
            for trial_num, df in trial.items():
                x = 0
                w_idx = np.where(df.event.isin(self.params['well_events']))[0]
                e_idx = np.where(df.event.isin(self.params['eating_events']))[0]
                s_idx = np.where(df.event.isin(['spont', 'Spont']))[0]
                # if s_idx.size > 0:
                #     spont_idx = df.neuron.iloc[s_idx[0]]
                # first_eating_time = df.neuron.iloc[e_idx[0]]
                well_times = df.neuron.iloc[w_idx]
                eating_times = df.neuron.iloc[e_idx]
                # last_well_time = df.neuron.iloc[w_idx[-1]]
                pre_well = df.loc[df.neuron < well_times.iloc[0], 'event']
                if pre_well.isin(self.params['eating_events']).any():
                    x = 1
                if pre_well.isin(self.params['eating_events']).any():
                    x = 1
                if x != 1:
                    yield df, w_idx, e_idx, s_idx