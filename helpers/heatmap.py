# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import rcParams

rcParams.update(
    {
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "xtick.major.width": "1.3",
        "axes.facecolor": "w",
        "axes.labelsize": 10,
        "lines.linewidth": 1,
    }
)

def get_largest_size(df):
    well_ts_diffs = df['well_ts'].apply(lambda x: max(np.array(x)) - min(np.array(x)) if len(np.array(x)) > 0 else 0)
    eat_ts_diffs = df['eat_ts'].apply(lambda x: max(np.array(x)) - min(np.array(x)) if len(np.array(x)) > 0 else 0)
    size = max(max(well_ts_diffs), max(eat_ts_diffs))  # max difference in timestamps
    return size


def get_largest_size_stretch(df, column):
    ts_diffs = df[column].apply(lambda x: max(np.array(x)) - min(np.array(x)) if len(np.array(x)) > 0 else 0)
    size = max(ts_diffs)  # max difference in timestamps
    return size


def sf_heatmap(df, binsize=0.1, padding=1, cmap='plasma', sigma=0):
    # Get the largest size for bins
    largest_size_of_interval = get_largest_size(df)

    # Prepare subplots
    fig, axs = plt.subplots(1, 2, sharey=True, figsize=(20, 10))

    vmin = np.inf  # Initialize minimum value
    vmax = -np.inf  # Initialize maximum value
    data_list = []  # Will hold all our data

    for index, column in enumerate(['well_ts', 'eat_ts']):
        allhist = []
        all_data = np.empty((0, int(largest_size_of_interval / binsize)))  # Initialized to empty array

        for idx, row in df.iterrows():
            # bin the spikes in the interval,
            ts = row[column]  # timestamps

            # Ensure ts is not empty
            hist = np.histogram(ts, bins=np.arange(min(ts), max(ts), binsize))[0]
            hist = gaussian_filter(hist, sigma)
            if len(hist) > largest_size_of_interval:
                # if the interval is larger than the graph
                hist = hist[:int(largest_size_of_interval / binsize)]
            while len(hist) < int(largest_size_of_interval / binsize):
                # if the interval is too short then extend it with nans
                hist = np.append(hist, np.nan)
            allhist.append(hist)

        container = np.empty((len(allhist), int(largest_size_of_interval / binsize)))
        container[:] = np.nan
        for i in range(len(allhist)):
            container[i, :] = allhist[i]
        all_data = np.concatenate((all_data, container), 0)
        test = np.empty((padding, int(largest_size_of_interval / binsize)))  # add some empty lines for visualization
        test[:] = np.nan
        all_data = np.concatenate((all_data, test), 0)
        data_list.append(all_data)

        # Update vmin and vmax
        vmin = min(vmin, np.nanmin(all_data))
        vmax = max(vmax, np.nanmax(all_data))

    for index, data in enumerate(data_list):
        ax = axs[index]
        plot = sb.heatmap(data, cmap=cmap, robust=True, ax=ax, vmin=vmin, vmax=vmax, cbar=False)
        plot.axes.get_yaxis().set_visible(False)
        ticks = np.arange(0, data.shape[1] + 1, 1 / binsize)
        plot.axes.get_xaxis().set_ticks(ticks)
        plot.axes.get_xaxis().set_ticklabels(ticks * binsize)
        plt.xlabel('Time (s)')

        if index == 0:  # Flip the x-axis of the left heatmap
            ax.invert_xaxis()

    fig.subplots_adjust(wspace=0.13)  # Reduce space between subplots
    # Transforms to figure coordinates
    inv = fig.transFigure.inverted()

    # Place labels to the right of the first heatmap
    for index, label in enumerate(df['event']):
        # Transform data coordinate to figure coordinate
        x_fig, y_fig = inv.transform(axs[0].transData.transform([largest_size_of_interval, index]))
        plt.text(x_fig + 0.05, y_fig - 0.02, label, transform=fig.transFigure)

    # Create a colorbar on the right
    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.78])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cbar_ax, label='spikes/bin')
    plt.show()

def sf_heatmap_stretch(df, binsize=0.1, padding=1, cmap='plasma', sigma=1):
    largest_size_of_intervals = {'well_ts': get_largest_size_stretch(df, 'well_ts'), 'eat_ts': get_largest_size_stretch(df, 'eat_ts')}
    fig, axs = plt.subplots(1, 2, figsize=(20, 10))
    vmin = np.inf
    vmax = -np.inf
    data_list = []

    for index, column in enumerate(['well_ts', 'eat_ts']):
        largest_size_of_interval = largest_size_of_intervals[column]
        allhist = []
        all_data = np.empty((0, int(largest_size_of_interval / binsize)))

        for idx, row in df.iterrows():
            ts = row[column]
            hist = np.histogram(ts, bins=np.arange(min(ts), max(ts), binsize))[0]
            hist = gaussian_filter(hist, sigma)
            if len(hist) > largest_size_of_interval:
                hist = hist[:int(largest_size_of_interval / binsize)]
            while len(hist) < int(largest_size_of_interval / binsize):
                hist = np.append(hist, np.nan)
            allhist.append(hist)

        container = np.empty((len(allhist), int(largest_size_of_interval / binsize)))
        container[:] = np.nan
        for i in range(len(allhist)):
            container[i, :] = allhist[i]
        all_data = np.concatenate((all_data, container), 0)
        test = np.empty((padding, int(largest_size_of_interval / binsize)))
        test[:] = np.nan
        all_data = np.concatenate((all_data, test), 0)
        data_list.append(all_data)
        vmin = min(vmin, np.nanmin(all_data))
        vmax = max(vmax, np.nanmax(all_data))

    for index, data in enumerate(data_list):
        ax = axs[index]
        plot = sb.heatmap(data, cmap=cmap, robust=True, ax=ax, vmin=vmin, vmax=vmax, cbar=False)
        plot.axes.get_yaxis().set_visible(False)
        ticks = np.arange(0, data.shape[1] + 1, 2 / binsize)  # plotting every other tick
        plot.axes.get_xaxis().set_ticks(ticks)
        plot.axes.get_xaxis().set_ticklabels(ticks * binsize)
        plt.xlabel('Time (s)')
        if index == 0:
            ax.invert_xaxis()

    inv = fig.transFigure.inverted()

    for index, label in enumerate(df['event']):
        x_fig, y_fig = inv.transform(axs[0].transData.transform([largest_size_of_intervals['well_ts'], index]))
        plt.text(x_fig + 0.06, y_fig - 0.02, label, transform=fig.transFigure)

    fig.subplots_adjust(wspace=0.12)

    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.78])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    fig.colorbar(sm, cbar_ax, label='spikes/bin')
    plt.show()

