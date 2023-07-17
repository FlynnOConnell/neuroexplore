# -*- coding: utf-8 -*-
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from matplotlib import rcParams
from pathlib import Path
import matplotlib as mpl

rcParams.update(
    {
        # colorbar title and labels size 14
        "font.size": 14,
        "axes.titlesize": 14,
        "axes.labelsize": 14,
        "axes.labelpad": 10,
        "font.weight": "bold",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.5,
        "xtick.labelsize": 14,
        "xtick.major.width": "1.5",
        "lines.linewidth": 1,
    }
)

def get_spont_mean(spont_ts_list, binsize):
    total_bins = 0
    total_spikes = 0
    for spont_ts in spont_ts_list:
        if len(spont_ts) > 0:
            spont_hist = np.histogram(spont_ts, bins=np.arange(min(spont_ts), max(spont_ts), binsize))[0]
            total_bins += len(spont_hist)
            total_spikes += spont_hist.sum()
    return total_spikes / total_bins if total_bins > 0 else 0


def get_largest_size_stretch(df, column):
    ts_diffs = df[column].apply(lambda x: max(np.array(x)) - min(np.array(x)) if len(np.array(x)) > 0 else 0)
    size = max(ts_diffs)  # max difference in timestamps
    return size


def sf_heatmap_stretch(df, spont_ts, binsize=0.1, padding=1, cmap='coolwarm', sigma=1, filename=None, save=False,
                       cutoff=None, **kwargs):

    def time_diff(df_row):
        return df_row['eat_ts'][-1] - df_row['eat_ts'][0]

    df = df[df['eat_ts'].map(lambda x: len(x) > 0)]
    df = df.sort_values('event')
    df = df[df.apply(time_diff, axis=1) <= 50]

    largest_size_of_intervals = {'well_ts': get_largest_size_stretch(df, 'well_ts'),
                                 'eat_ts': get_largest_size_stretch(df, 'eat_ts')}
    if cutoff:
        largest_size_of_intervals = {'well_ts': min(cutoff, largest_size_of_intervals['well_ts']),
                                     'eat_ts': min(cutoff, largest_size_of_intervals['eat_ts'])}

    fig, axs = plt.subplots(1, 2, figsize=(20, 10),)
    data_list = []
    spont_mean = get_spont_mean(spont_ts, binsize)

    all_values = []  # Placeholder for all values after subtracting spont_mean

    for index, column in enumerate(['well_ts', 'eat_ts']):
        largest_size_of_interval = largest_size_of_intervals[column]
        allhist = []
        all_data = np.empty((0, int(largest_size_of_interval / binsize)))

        for idx, row in df.iterrows():
            ts = row[column]
            if len(ts) == 0:
                continue
            hist = np.histogram(ts, bins=np.arange(min(ts), max(ts), binsize))[0]
            hist = gaussian_filter(hist, sigma)
            hist = hist - spont_mean

            all_values.extend(hist)  # Store values after subtracting spont_mean

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

    # Calculate 1st and 99th percentile
    vmin = np.percentile(all_values, 1)
    vmax = np.percentile(all_values, 99)

    for index, data in enumerate(data_list):
        ax = axs[index]
        # Use vmin and vmax for color normalization
        plot = sb.heatmap(data, cmap=cmap, ax=ax, vmin=vmin, vmax=vmax, cbar=False)
        plot.axes.get_yaxis().set_visible(False)

        # calculate the additional bins needed for the 1 unit extension
        extension_bins = int(1 / binsize)
        # extend the x-axis by 1 unit (converted to bins)
        ticks = np.arange(0, data.shape[1] + extension_bins, 1 / binsize)

        plot.axes.set_xticks(ticks)
        plot.axes.set_xticklabels(ticks * binsize, rotation=60)

        if index == 0:
            ax.invert_xaxis()

    inv = fig.transFigure.inverted()
    for index, label in enumerate(df['event']):
        x_fig, y_fig = inv.transform(axs[0].transData.transform([largest_size_of_intervals['well_ts'], index]))
        plt.text(x_fig + 0.05, y_fig - 0.022, label, transform=fig.transFigure)

    fig.subplots_adjust(wspace=0.12)

    cbar_ax = fig.add_axes([0.92, 0.12, 0.02, 0.78])
    norm = plt.Normalize(vmin=vmin, vmax=vmax)
    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    cbar = fig.colorbar(sm, cbar_ax)

    # Change colorbar labels to percentages
    cbar_labels = [item.get_text() for item in cbar.ax.get_yticklabels()]
    new_labs = np.linspace(-1, +1, len(cbar_labels)).tolist()
    new_labs = np.round(new_labs, 1)
    cbar.ax.set_yticklabels(new_labs)
    cbar.set_label('Spks/100ms bin (Fraction of Spontaneous)', rotation=270, labelpad=20, fontsize=14)

    # save the figure to data/res/heatmaps/filename.png
    title = Path(filename).stem
    title = title.replace('_', ' ')
    plt.figtext(0.6, 0.95, title, ha='right', va='top', fontsize=20, fontweight='bold')
    if save:

        plt.savefig(str(filename) + '.png', dpi=300, bbox_inches='tight', )
        print('Figure saved')
    else:
        plt.show()
