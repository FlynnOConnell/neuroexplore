import numpy as np
import pandas as pd
from pandas import DataFrame
from scipy import stats, signal
import sys
import os

from nex import nexfile

curr_file = "SFN14_2018-12-07_vision_SF.nex"

__dir = os.getcwd()

# File to analyze
__datafile = __dir + "/data/" + curr_file

__reader = nexfile.Reader(useNumpy=True)
data = __reader.ReadNexFile(__datafile)

# indicate the names of the intervals which you want included in the heatmap
sf_labels = (
    "Spont",
    "Well (Empty)",
    "Well (Peanut)",
    "Well (Apply)",
    "Well (M. Chocolate)",
    "Eat (Apple)",
    "Eat (M. Chocolate)",
    "Eat (Apple)",
)

sf_events = (
    "Spont",
    "BLW_bout_Int",
    "FLW_bout_Int",
    "FRW_bout_Int",
    "BRW_bout_Int",
    "e_FR",
    "e_BR",
    "e_FL",
)
binsize = .1  # binsize for binning spikes (seconds)
end = 6  # how long after interval start to display (Seconds)
start = -1  # how long before the interval start to display (seconds)
padding = 1  # space between each event (vertically) on the graph
ts_adjust = 0.005  # time shift to adjust for the time it takes to lick
timestamps = {}

neurons = {}
trial_times = {}
eat_events = {}
intervals = {}
timestamps["Start"] = 0.0

# populate data containers
for var in data['Variables']:
    if var['Header']['Type'] == 2 and var['Header']['Name'] in sf_events:
        intervals[var['Header']['Name']] = var['Intervals']
    if var['Header']['Type'] == 0:
        neurons[var['Header']['Name']] = var['Header']['Name']
        timestamps[var['Header']['Name']] = var['Timestamps']

to_del = ["RSTART", "RSTOP", "Solid Food", "Strobed"]

# for key in to_del:
#     timestamps.pop(key)
df: DataFrame = pd.DataFrame()
neuron = neurons["SPK08a"]
neurons[neuron] = eat_events
# intialize data containers
neur = timestamps[neuron]
df = pd.DataFrame()
df_final = pd.DataFrame()
eat_events = dict.fromkeys([
    'Spont',
    'BLW_bout_Int',
    'BRW_bout_Int',
    'FLW_bout_Int',
    'FRW_bout_Int',
    'e_FR',
    'e_BR',
    'e_FL',
])

for key in eat_events:
    intervs = list(zip(intervals[key][0], intervals[key][1]))
    allhist = []
    mean_std = []
    for interval in intervs:
        if interval[1] - interval[0] >= 1:  # ignore intervals smaller than 1 second
            # bin the spikes in the interval
            hist = np.histogram(neur, bins=np.arange(interval[0] + start, interval[1], binsize))[0]
            if len(hist) > int((end - start) / binsize):
                # if the interval is larger than the graph
                hist = hist[0:int((end - start) / binsize)]
            while len(hist) < int((end - start) / binsize):
                # if the interval is too short then extend it with nans
                hist = np.append(hist, np.nan)
            allhist.append(hist)
    # Get the data container ready for all the histogram data and insert the data
    container = np.empty((len(allhist), int((end - start) / binsize)))
    container[:] = np.nan
    for i in range(len(allhist)):
        container[i, :] = allhist[i]
    # store all data for each event in a dictionary
    eat_events[key] = container
    # calculate mean for each event in the dictionary
    if 'Spont' in eat_events:
        key_mean = np.nanmean(container) * 10
        key_std = np.nanstd(container)
        mean_std.append(key_mean)
        mean_std.append(key_std)
        mean_std.append(len(allhist))
        eat_events[key] = mean_std
    else:
        key_array = np.empty([len(allhist), 15])
        a = np.array(eat_events[key])
        for i in range(10, 25):
            key_array[:, i - 10] = a[:, i]
        key_mean = np.nanmean(key_array) * 10
        key_std = np.nanstd(key_array)
        mean_std.append(key_mean)
        mean_std.append(key_std)
        mean_std.append(len(allhist))
        eat_events[key] = mean_std
df = pd.DataFrame.from_dict(eat_events, orient='index', columns=['MEAN', 'SD', 'TRIALS'])
# neurons[neuron]=eat_events
df.insert(loc=3, column='neuron', value=neuron)

# Initializing data containers
intervals = {}
neurons = ()
timestamps = {}

# populate data containers
for var in data['Variables']:
    if var['Header']['Type'] == 2 and var['Header']['Name'] in sf_events:
        intervals[var['Header']['Name']] = var['Intervals']
    if var['Header']['Type'] == 0:
        neurons = neurons + tuple([var['Header']['Name']])
        timestamps[var['Header']['Name']] = var['Timestamps']

for neuron in neurons:
    # intialize data containers
    neur = timestamps[neuron]
    allhist = []
    heatmap_data = {}
    lengths = [0]
    all_data = np.empty((0, int((end - start) / binsize)))
    for key in sf_events:  # for each event
        # get list of intervals
        intervs = list(zip(intervals[key][0], intervals[key][1]))
        allhist = []
        for interval in intervs:
            if interval[1] - interval[0] >= 1:  # ignore intervals smaller than 1 second
                # bin the spikes in the interval
                hist = np.histogram(neur, bins=np.arange(interval[0] + start, interval[1], binsize))[0]
                if len(hist) > int((end - start) / binsize):
                    # if the interval is larger than the graph
                    hist = hist[0:int((end - start) / binsize)]
                while len(hist) < int((end - start) / binsize):
                    # if the interval is too short then extend it with nans
                    hist = np.append(hist, np.nan)
                allhist.append(hist)
        # Get the data container ready for all the histogram data and insert the data
        container = np.empty((len(allhist), int((end - start) / binsize)))
        container[:] = np.nan
        for i in range(len(allhist)):
            container[i, :] = allhist[i]
        all_data = np.concatenate((all_data, container), 0)
        test = np.empty((padding, 70))  # add some empty lines for visualization
        test[:] = np.nan
        all_data = np.concatenate((all_data, test), 0)
        lengths.append(all_data.shape[0] + padding)  # store lengths for adding varnames

    # variable name positions
    varpos = [.5 * (lengths[i] + lengths[i + 1]) for i in range(len(lengths) - 1)]

temp = []

