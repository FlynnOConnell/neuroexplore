
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
import os
from nex import nexfile

from helpers.funcs import get_matched_time

file = "SFN14_2018-12-07_vision_SF.nex"
savepath = "/Users/flynnoconnell/Pictures/sf"

x_datafile = os.getcwd() + "/data/" + file
x_reader = nexfile.Reader(useNumpy=True)

data = x_reader.ReadNexFile(x_datafile)
# %%
well_events = ['BLW_bout_Int',
               'FLW_bout_Int',
               'FRW_bout_Int',
               'BRW_bout_Int']
eating_events = ['e_FR',
                 'e_BR',
                 'e_FL',
                 'e_BL']

colors = {key: "blue" for key in well_events}
colors['e_FR'] = 'brown'
colors['e_BR'] = 'orange'
colors['e_FL'] = 'pink'
colors['e_BL'] = 'red'
colors['Spont'] = 'gray'

sf_events = well_events + eating_events + ['Spont']

binsize = .1  # binsize for binning spikes (seconds)
end = 6  # how long after interval start to display (Seconds)
start = -1  # how long before the interval start to display (seconds)

# Data containers
neurons = ()
timestamps, intervals = {}, {}
well_intervals, eating_intervals, spont_intervals = {}, {}, {}
all_data = {}

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

ts_holder = defaultdict(list)
for neuron in neurons:
    # intialize data containers
    neur = timestamps[neuron]
    all_data[neuron] = neuro_df = pd.DataFrame(columns=['Neuron', 'Event', 'Color'])
    neuro_df["Neuron"] = neur
    for key in intervals:
        intervs = list(zip(intervals[key][0], intervals[key][1]))
        for interval in intervs:
            # hist = np.histogram(neur, bins=np.arange(interval[0] + start, interval[1], binsize))[0]
            idx = np.where((neuro_df['Neuron'] >= interval[0]) & (neuro_df['Neuron'] <= interval[1]))[0]
            neuro_df.loc[idx[0]:idx[-1], 'Event'] = key
            neuro_df.loc[idx[0]:idx[-1], 'Color'] = colors[key]

neuro_df.Color = neuro_df.Color.fillna('black')

trials = {
    k: {} for k in eating_events
}
counter = 0
x = 0

for row in neuro_df.iterrows():
    if (row[1]['Event'] in well_events) & (x in [0, 2]):
        start = row[1]['Neuron']
        begin = start - 2
        x = 1
        well_spike_count = 0
    if (row[1]['Event'] in well_events) & (x == 1):
        well_spike_count += 1
        event_tracker = row[1]['Event']
        if type(neuro_df.loc[row[0] + 1, 'Event']) == float:
            num_well_ev = well_spike_count
            x = 2
    if row[1]['Event'] in eating_events:
        if type(neuro_df.loc[row[0] + 1, 'Event']) == float:
            end = row[1]['Neuron']
            x = 0
            counter += 1
            idx = np.where((neuro_df['Neuron'] >= begin) & (neuro_df['Neuron'] <= end))[0]
            trials[row[1]['Event']][counter] = neuro_df.loc[idx[0]:idx[-1], :]

test = trials['e_BR'][4]

# %%

bins = np.arange(test.Neuron.iloc[0], test.Neuron.iloc[-1] - test.Neuron.iloc[0], binsize)
# plt.hist(test.iloc[:, 0], histtype='step')
y, x = np.histogram(test.Neuron, bins.reshape(-1))
plt.bar(x[:-1], y, width=binsize)
plt.show()

# variable name positions
# varpos = [.5 * (lengths[i] + lengths[i + 1]) for i in range(len(lengths) - 1)]

# generate plot
fig, ax = plt.subplots()

# plot = sb.heatmap(test['Neuron'], cmap='jet', cbar_kws={'label': 'spikes/bin'})
# ax.get_yaxis().set_visible(False)
# ax.get_xaxis().set_ticks(np.arange(0, test.shape[1] + 1, 1 / binsize))
#
# ax.get_xaxis().set_ticklabels(np.arange(test.iloc[0, 0], test.iloc[test.shape[0]-2], 1))
# plt.xticks(rotation=0)
# plt.axvline(x=test.iloc[0, 0]+2 / binsize, ymin=0, ymax=1, color='red')
# plt.xlabel('Time (s)')
plt.tight_layout()
plt.show()

temp = []
