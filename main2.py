from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams, lines
import os
from nex import nexfile

from helpers import funcs, ax_helpers

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
neuro_df = pd.DataFrame()
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

trials = {k: {} for k in eating_events}

# %%

counter = 0
x = 0
for row in neuro_df.iterrows():
    ts_idx = row[0]
    ts_stamp = row[1]['Neuron']
    ts_event = row[1]['Event']

    if (ts_event in well_events) & (x in [0, 2]):
        start = ts_stamp
        begin = start - 2
        x = 1
        well_spike_count = 0
    if (ts_event in well_events) & (x == 1):
        well_spike_count += 1
        event_tracker = ts_event
        if type(neuro_df.loc[ts_idx + 1, 'Event']) == float:
            num_well_ev = well_spike_count
            x = 2
    if ts_event in eating_events:
        if type(neuro_df.loc[row[0] + 1, 'Event']) == float:
            end = ts_stamp
            x = 0
            counter += 1
            idx = np.where((neuro_df['Neuron'] >= begin) & (neuro_df['Neuron'] <= end))[0]
            trials[ts_event][counter] = neuro_df.loc[idx[0]:idx[-1], :]

# %%

rcParams.update(
    {
        "font.weight": "bold",
        "font.family": "Arial",
        'font.sans-serif': "Arial",
        "axes.labelweight": "bold",
        "xtick.major.width": "1.3",
        "axes.facecolor": "w",
        "axes.labelsize": 10,
        "lines.linewidth": 1,
    }

)

a_trial = trials['e_BR'][4]
bins = np.arange(a_trial.Neuron.iloc[0], a_trial.Neuron.iloc[-1], binsize)
# plt.hist(test.iloc[:, 0], histtype='step')
y_height, x_bins = np.histogram(a_trial.Neuron, bins.reshape(-1))
fig, ax = plt.subplots()
barcol = [funcs.get_matched_time(np.asarray(a_trial.Neuron), x)[0] for x in x_bins]

a_ev = a_trial.Color.iloc[[np.where(a_trial.Neuron == x)[0][0] for x in barcol]]

thisdict = {'Peanuts': 'orange',
            'Well': 'blue',
            'Movement': 'black'
            }
ax = ax_helpers.get_legend(ax, thisdict, a_ev, 1)

ax.bar(x_bins[:-1], y_height, width=binsize, color=a_ev)
ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1)

# ax.hlines(y=ax.get_ylim()[1], xmin=4, xmax=20, linewidth=2, color='r')

plt.show()

temp = []
