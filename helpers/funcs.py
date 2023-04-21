# -*- coding: utf-8 -*-
"""
#funcs.py
Module(util): General getter/setter/checker functions.
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Iterable, Optional, Any, Tuple
import matplotlib
from matplotlib import rcParams, lines
import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# %% COLLAPSE DATA STRUCTURES

def filter_dict(my_dict: dict, to_filter: Iterable):
    """Filter dictionary based on a list of colors."""
    return {k: v for k, v in my_dict.items() if my_dict[k] in np.unique(to_filter)}


def check_numeric(my_str: str):
    """ Return boolean True if string is all numbers, otherwise False."""
    return my_str.isdecimal()


def check_path(my_str: str | Path):
    """ Return boolean True if string is a path, otherwise False."""
    return isinstance(my_str, Path) or any(x in my_str for x in ["/", "\\"])


def keys_exist(element, *keys):
    """Check if *keys (nested) exists in `element` (dict)"""
    if len(keys) == 0:
        raise AttributeError("keys_exists() expects at least two arguments, one given.")
    _element = element
    for key in keys:
        try:
            _element = _element[key]
        except KeyError:
            return False
    return True


def iter_events(event_dct, gap: int = 5):
    """
    Given an interval 'gap',
    iterate through a dictionary and generate an interval (start, stop) return value.
    """
    for event, ts in event_dct.items():
        intervals = interval(ts, gap)
        interv: Iterable
        for interv in intervals:
            yield event, interv


def flatten(lst: Iterable) -> list:
    return [item for sublist in lst for item in sublist]


def check_unique_path(path) -> str:
    if hasattr(path, 'stem'):
        path = path.__str__()
    filename, extension = os.path.splitext(path)
    counter = 1
    while os.path.exists(path):
        path = filename + "" + str(counter) + "" + extension
        counter += 1
    return path

def interval(
        lst: Iterable[any], gap: Optional[int] = 1, outer: bool = False
) -> list[tuple[Any, Any]]:
    """
    Create intervals where there elements are separated by either:
        -less than gap.
        -more than gap.
    Args:
        lst (Iterable): Iterable to search.
        gap (int | float): length of interval.
        outer (bool): Makes larger than (gap) intervals.
    Returns:
         interv (list): New list with created interval.
    """
    interv, tmp = [], []

    for v in lst:
        if not tmp:
            tmp.append(v)
        elif abs(tmp[-1] - v) < gap:
            tmp.append(v)
        elif outer:
            interv.append(tuple((tmp[-1], v)))
            tmp = [v]
        else:
            interv.append(tuple((tmp[0], tmp[-1])))
            tmp = [v]
    return interv

def get_peak_window(time: Iterable[any], peak: float) -> list:
    """
    Returns the index of tracedata centered 1s around the peak flourescent value for
    that trial.
    Args:
        time (list | pd.Series): List of all time values.
        peak (float) : peak time
    Returns:
         window_ind (list): list of index values to match time.
    """
    time: Iterable[any]
    aux, window_ind = [], []
    for valor in time:
        aux.append(abs(peak - valor))
    window_ind.append(aux.index(min(aux)) - 20)
    window_ind.append(aux.index(min(aux)) + 20)
    return window_ind

def get_matched_time(time: np.ndarray, match: np.ndarray | int) -> np.ndarray:
    """
    Finds the closest number in time to the input. Can be a single value,
    or list. Finds all absolute differences between match and time, find
    the minima of these abs. values and the argument value of the minima. Return
    all matches.
    Args:
        time : np.ndarray | int
            Correct values to be matched to.
        match : Iterable[any]
            Values to be matched.
    Returns
         np.ndarray of matched times.
    """
    time = np.asarray(time)
    match = np.asarray(match).reshape(-1, 1)
    mins = np.argmin(np.abs(match - time), axis=1)
    return np.array([time[mins[i]] for i in range(len(match))])

def vec_to_sq(vec: np.ndarray):
    vec = vec.astype('float')
    sq = int(np.ceil(np.sqrt(vec.size)))
    to_pad = (sq * sq) - vec.size
    if to_pad % 2:
        return np.pad(vec, (to_pad/2, to_pad/2), mode='constant', constant_values=np.nan).reshape((sq, sq))
    else:
        return np.pad(vec, (to_pad-1, 1), mode='constant', constant_values=np.nan).reshape((sq, sq))

def discrete_mean(arr):
    """Calculate the mean along the axis of a discrete array."""
    arr = np.asarray(arr)
    cumsum = np.cumsum(arr)
    cumsum[2:] = cumsum[2:] - cumsum[:-2]
    return np.asarray(cumsum[2 - 1:] / 2, dtype=int)

def rolling_average_std(array, window_size=10):
    """
    Calculates the rolling average and standard deviation of an input numpy array
    with a window size of 10, where each interval overlaps the previous interval by
    all but 1 array point.

    Args:
    array (numpy array): Input array for which rolling average and standard deviation
                         need to be calculated.
    window_size (int): Size of the window for calculating rolling average and standard

    Returns:
    tuple: A tuple containing two numpy arrays: rolling averages and rolling standard
           deviations.
    """
    num_points = len(array)
    # Extend the array to ensure the last interval has at least 10 values
    if num_points % window_size != 0:
        num_extend = window_size - (num_points % window_size) + 1
        array = np.pad(array, (0, num_extend), mode='edge')

    # Calculate rolling averages and rolling standard deviations
    rolling_averages, rolling_std = [], []
    for i in range(0, num_points, window_size - 1):
        interval = array[i:i + window_size]
        average = np.mean(interval)
        std = np.std(interval)
        rolling_averages.append(average)
        rolling_std.append(std)

    return np.array(rolling_averages), np.array(rolling_std)

def calculate_window_stats(arr):
    """
    Calculate rolling window statistics (average and standard deviation) for a given NumPy array.

    Args:
    - arr (np.ndarray): Input array

    Returns:
    - avg (np.ndarray): Array of window averages
    - std (np.ndarray): Array of window standard deviations
    """
    window_size = 10
    arr_len = len(arr)
    avg = np.empty(arr_len)
    std = np.empty(arr_len)
    for i in range(arr_len):
        if i + window_size <= arr_len:
            window = arr[i:i + window_size]
        else:
            window = arr[i:]
        if len(window) >= 2:
            avg[i] = np.mean(window)
            if len(window) >= 3:
                std[i] = np.std(window)
            else:
                std[i] = np.nan
        else:
            avg[i] = np.nan
            std[i] = np.nan
    return avg, std
