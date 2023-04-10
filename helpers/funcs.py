# -*- coding: utf-8 -*-
"""
#funcs.py
Module(util): General getter/setter/checker functions.
"""
from __future__ import annotations
import os
import logging
from pathlib import Path
from typing import Iterable, Optional, Any

import numpy as np

from .wrappers import typecheck

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(message)s")


# %% COLLAPSE DATA STRUCTURES

def filter_dict(my_dict: dict, to_filter: Iterable):
    """Filter dictionary based on a list of colors."""
    return {k: v for k, v in my_dict.items() if my_dict[k] in np.unique(to_filter)}


@typecheck(str)
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


@typecheck(dict, int)
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


@typecheck(Iterable)
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


@typecheck(Iterable, int)
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


@typecheck(Iterable[any], Iterable[any])
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


def get_matched_time(time: np.ndarray, match: np.ndarray) -> np.ndarray:
    """
    Finds the closest number in time to the input. Can be a single value,
    or list. Find all the absolute differences between match and time, find
    the minima of these abs. values and the argument value of the minima. Return
    all matches.
    Args:
        time : np.ndarray
            Correct values to be matched to.
        match : Iterable[any]
            Values to be matched.
    Returns
         np.ndarray of matched times.
    """
    match = np.asarray(match).reshape(-1, 1)
    mins = np.argmin(np.abs(match - time), axis=1)
    return np.array([time[mins[i]] for i in range(len(match))])

def spike_frequency(spikes: np.ndarray) -> np.ndarray:
    """
    Frequency, as f = n/T
    Args:
        spikes : np.ndarray
            Array of spike times in a constant timespan.

    Returns
         Frequency of spike train in spikes/second.
    """

    return np.array([time[mins[i]] for i in range(len(match))])