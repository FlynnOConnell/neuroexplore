from collections import defaultdict, OrderedDict
import operator
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os
from nex import nexfile
from helpers.funcs import get_matched_time

file = "SFN14_2018-12-07_vision_SF.nex"
savepath = "/Users/flynnoconnell/Pictures/sf"

__dir = os.getcwd()
__datafile = __dir + "/data/" + file
__reader = nexfile.Reader(useNumpy=True)

data = __reader.ReadNexFile(__datafile)

well_events = ['BLW_bout_Int',
               'FLW_bout_Int',
               'FRW_bout_Int',
               'BRW_bout_Int']
eating_events = ['e_FR',
                 'e_BR',
                 'e_FL',
                 'e_BL']

sf_events = well_events + eating_events + ['Spont']

binsize = .1  # binsize for binning spikes (seconds)
end = 6  # how long after interval start to display (Seconds)
start = -1  # how long before the interval start to display (seconds)

# Data containers
neurons = ()
timestamps, intervals = {}, {}
well_intervals, eating_intervals, spont_intervals = {}, {}, {}

# populate data containers
for var in data['Variables']:
    if var['Header']['Type'] == 2 and var['Header']['Name'] in well_events:
        intervals[var['Header']['Name']] = var['Intervals']
        well_intervals[var['Header']['Name']] = var['Intervals']
    if var['Header']['Type'] == 2 and var['Header']['Name'] in eating_events:
        intervals[var['Header']['Name']] = var['Intervals']
        eating_intervals[var['Header']['Name']] = var['Intervals']
    if var['Header']['Type'] == 2 and var['Header']['Name'] in ['Spont']:
        intervals[var['Header']['Name']] = var['Intervals']
        spont_intervals[var['Header']['Name']] = var['Intervals']
    if var['Header']['Type'] == 0:
        neurons = neurons + tuple([var['Header']['Name']])
        timestamps[var['Header']['Name']] = var['Timestamps']

seq = pd.DataFrame(columns=['Event', 'Trial', 'Data'])


for wellev, (interval_start, interval_stop) in well_intervals.items():




ts_holder = defaultdict(list)
bin_holder = defaultdict(list)
for neuron in neurons:
    # intialize data containers
    neur = timestamps[neuron]
    for key in eating_intervals:
        intervs = list(zip(intervals[key][0], intervals[key][1]))
        for interval in intervs:
            if interval[1] - interval[0] >= 1:
                start_eating = interval[0]
                for well_ev in well_intervals:
                    well_ivl = list(zip(intervals[well_ev][0], intervals[well_ev][1]))
                    for well_int in well_ivl:
                        end_well = well_int[1]
                        if start_eating - end_well <= 4:
                            print(key, well_ev)






temp = []
