import numpy as np
from pathlib import Path
import seaborn as sb
import matplotlib.pyplot as plt
import pandas as pd
from data_utils.file_handling import get_nex
from helpers import plot as sfplot
from helpers.plot import freq_histogram_col

path = Path.home() / "Dropbox/Lab/SF/nexfiles/SFN11_2018-08-16_SF.nex"
data = get_nex(path)

binsize = .1  # binsize for binning spikes (seconds)
# end = 6  # how long after interval start to display (Seconds)
start = -2  # how long before the interval start to display (seconds)
padding = 4  # space between each event (vertically) on the graph

# Initializing data containers
intervals = {}  # store the original sf events timeframe data from nex(scored vedio)
neurons = ()
timestamps = {}  # store timestamp for different neurons
eat_events = {}
sf_events = ([
    'Spont',
    'BLW_bout_Int',
    'BRW_bout_Int',
    'FLW_bout_Int',
    'FRW_bout_Int',
    'e_BR',
    'e_FR',
    'e_FL',
    'e_BL',
    # 'e_a',
    # 'e_b',
    # 'e_c',
    # 'e_n',
    'grooming'
])
counts = {}
# populate data containers
for var in data['Variables']:
    counts[var['Header']['Name']] = var['Header']['Count']
    if var['Header']['Type'] == 2 and var['Header']['Name'] in sf_events:
        intervals[var['Header']['Name']] = list(zip(var['Intervals'][0], var['Intervals'][1]))
    if var['Header']['Type'] == 0:
        neurons = neurons + tuple([var['Header']['Name']])
        timestamps[var['Header']['Name']] = var['Timestamps']


neuron_timestamps = timestamps['SPK08a']
for event, intervs in intervals.items():
    for this_interval in intervs:
        if this_interval[1] - this_interval[0] >= 1:  # ignore intervals smaller than 1 second
            end = this_interval[1] - this_interval[0]
            spikes = neuron_timestamps[(neuron_timestamps > this_interval[0] + start) & (neuron_timestamps < this_interval[1])]
            spikes = spikes - this_interval[0]
            event_df = pd.DataFrame({'spikes': spikes, 'event': event})
            event_df['color'] = np.where(event_df['spikes'] < 0, 'blue', 'red')
            freq_histogram_col(event_df, title=event)


y = 2


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
        if key == 'Spont':
            intervs = intervs[7:]
        for interval in intervs:
            if interval[1] - interval[0] >= 1:  # ignore intervals smaller than 1 second
                if key in ['e_FL', 'FLW_bout_Int']:
                    spikes = neur[(neur > interval[0] + start) & (neur < interval[1])]
                    sfplot.freq_histogram(spikes, key, binsize)
                hist = np.histogram(neur, bins=np.arange(interval[0] + start, interval[1], binsize))[0]
                if len(hist) > int((end - start) / binsize):
                    # if the interval is larger than the graph
                    hist = hist[0:int((end - start) / binsize)]
                while len(hist) < int((end - start) / binsize):
                    # if the interval is too short then extend it with nans
                    hist = np.append(hist, np.nan)
                allhist.append(hist)

        container = np.empty((len(allhist), int((end - start) / binsize)))
        container[:] = np.nan
        for i in range(len(allhist)):
            container[i, :] = allhist[i]
        all_data = np.concatenate((all_data, container), 0)
        test = np.empty((padding, 70))
        test[:] = np.nan
        all_data = np.concatenate((all_data, test), 0)
        lengths.append(all_data.shape[0] + padding)  # store lengths for adding varnames

    # variable name positions
    varpos = [.5 * (lengths[i] + lengths[i + 1]) for i in range(len(lengths) - 1)]

    # generate plot
    plt.figure()
    plot = sb.heatmap(all_data, cmap='plasma', robust=False, cbar_kws={'label': 'spikes/bin'})
    plot.axes.get_yaxis().set_visible(False)
    plot.axes.get_xaxis().set_ticks(np.arange(0, all_data.shape[1] + 1, 1 / binsize))
    plot.axes.get_xaxis().set_ticklabels(np.arange(start, end + 1, 1))
    plt.xticks(rotation=0)
    plt.axvline(x=-start / binsize, ymin=0, ymax=1, color='red')
    plt.xlabel('Time (s)')
    plt.title(f'{neuron} - {path.name}')
    for i in range(len(sf_events)):
        plt.text(0, varpos[i], sf_events[i], ha='right')
    plt.tight_layout()
    plt.show()
    x = 5
    # plt.savefig(os.path.split(file)[0] + '/heatmap_{}_{}.png'.format(os.path.split(file)[1], neuron), dpi=300)
