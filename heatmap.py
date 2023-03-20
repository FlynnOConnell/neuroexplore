# -*- coding: utf-8 -*-
"""
Created on Tue Apr  6 13:18:56 2021

@author: Di Lorenzo Tech

This script will generate heatmaps for eating events. Just put your filename, and hit go!
No need to have neuroexplorer open for this

"""
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
import nex.nexfile as nex
import sys
import os

# File to analyze
file = r'R:\Iris\trial\SFN14_2019-03-29_SF.nex'

# indicate the names of the intervals which you want included in the heatmap
sf_events = (
    'Spont',
    'BLW_bout_Int',
    'FLW_bout_Int',
    'FRW_bout_Int',
    'BRW_bout_Int',
    'e_FR',
    'e_BR',
    'e_FL',
    'e_BL',
)

binsize = .1  # binsize for binning spikes (seconds)
end = 6  # how long after interval start to display (Seconds)
start = -1  # how long before the interval start to display (seconds)
padding = 1  # space between each event (vertically) on the graph

# Open NEXfile to get data
data = nex.Reader(useNumpy=True).ReadNexFile(file)

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
                hist = np.histogram(neur, bins=np.arange(interval[0] + start, interval[1],
                                                         binsize))[0]
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

    # generate plot
    plt.figure()
    plot = sb.heatmap(all_data, cmap='jet', robust=True, cbar_kws={'label': 'spikes/bin'})
    plot.axes.get_yaxis().set_visible(False)
    plot.axes.get_xaxis().set_ticks(np.arange(0, all_data.shape[1] + 1, 1 / binsize))
    plot.axes.get_xaxis().set_ticklabels(np.arange(start, end + 1, 1))
    plt.xticks(rotation=0)
    plt.axvline(x=-start / binsize, ymin=0, ymax=1, color='red')
    plt.xlabel('Time (s)')
    plt.title('File: {}\nNeuron: {}'.format(os.path.split(file)[1], neuron))
    for i in range(len(sf_events)):
        plt.text(0, varpos[i], sf_events[i], ha='right')
    plt.tight_layout()
    plt.savefig(
        os.path.split(file)[0] + '/heatmap_{}_{}.png'.format(os.path.split(file)[1],
                                                             neuron), dpi=300)
