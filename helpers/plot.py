import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams, lines
from helpers import funcs, ax_helpers

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

def freq_histogram(spikes, binsize: int | float = 0.1, legend_dict: dict = None) -> None:
    bins = np.arange(spikes.iloc[0], spikes.iloc[-1], binsize)
    y_height, x_bins = np.histogram(spikes, bins.reshape(-1))
    barcol = [funcs.get_matched_time(np.asarray(spikes), x)[0] for x in x_bins]
    spikes = spikes.reset_index(drop=True)
    a_ev = [[np.where(spikes == x)[0][0] for x in barcol]]

    fig, ax = plt.subplots()
    ax.bar(x_bins[:-1], y_height, width=.1, color=a_ev)
    ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1)
    if legend_dict:
        proxy, label = ax_helpers.get_handles_from_dict(legend_dict, 3, marker="s")
        ax.legend(
            handles=proxy,
            labels=label,
            facecolor=ax.get_facecolor(),
            edgecolor=None,
            fancybox=False,
            markerscale=3,
            shadow=True)

    ax.set_title("Frequency Histogram", fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Spikes / second", fontweight='bold')

    plt.tight_layout()
    plt.show()

def isi_histogram(spikes, colors, legend_dict: dict = None) -> None:
    fig, ax = plt.subplots()
    ax.bar(spikes.iloc[:-1], np.diff(spikes), width=0.1, color=colors)
    if legend_dict:
        proxy, label = ax_helpers.get_handles_from_dict(legend_dict, 3, marker="s")
        ax.legend(
            handles=proxy,
            labels=label,
            facecolor=ax.get_facecolor(),
            edgecolor=None,
            fancybox=False,
            markerscale=3,
            shadow=True)

    ax.set_title("Interspike Interval vs Time", fontweight='bold')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel("ISI", fontweight='bold')
    plt.tight_layout()
    plt.show()

def heatmap(plot_data):
    xlab = np.arange(-2, plot_data.shape[1]/10, 1)
    xloc = []
    fig, axs = plt.subplots()
    smoothed = gaussian_filter(plot_data, sigma=2)
    axs = sns.heatmap(smoothed, cmap='plasma', ax=axs, cbar_kws={'label': 'spk/s'})

    # Set tick positions for both axes
    axs.yaxis.set_ticks([i for i in range(plot_data.shape[0])])
    axs.xaxis.set_ticks([10, 20, 30])
    # Remove ticks by setting their length to 0
    axs.yaxis.set_tick_params(length=0)
    axs.xaxis.set_tick_params(length=0)

    # Remove all spines
    axs.set_frame_on(False)
    plt.show()

def trial_histogram(frame, sigma=1, width=0.1):
    fig, ax = plt.subplots()
    ax.bar(frame.neuron,gaussian_filter(frame.counts, sigma=sigma), width=width, color=frame.color)
    ax.set_title("Single Trial Histogram", fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (hz)", fontweight='bold')
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.show()
