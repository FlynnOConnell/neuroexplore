import signal
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams, lines
from scipy import signal
import os
from nex import nexfile
from data_reader import ReadData
from data_formatter import DataFormatter
from helpers import funcs, ax_helpers
import plot

# %%

file = "SFN16_2019-03-25_SF.nex"

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

savepath = "/Users/flynnoconnell/Pictures/sf"

_data = ReadData(file, os.getcwd() + "/data/")
data = DataFormatter(_data.raw_data, well_events, eating_events, colors)

data_trials = data.trials
df = data.df_sorted

tstamps = data.neurostamps
data.generate_trials_strict()
# average = [len(tstamps[np.where((interval[0] <= tstamps) & (tstamps <= interval[1]))]) / (interval[1] - interval[0]) for interval in data.interval_gen("Spont")]
my_list = []
x = 0
legend_dict = {
    "Spontaneous": "gray",
    "Transition": "black",
    "Well": "blue",
    "Eating": "orange"
}

from scipy.ndimage import gaussian_filter

proxy, label = ax_helpers.get_handles_from_dict(legend_dict, 3, marker="s")
for spikes, events, colors in data.generate_trials_strict():
    x += 1
    # fig, ax = plt.subplots()
    # ax.bar(spikes.iloc[:-1], np.diff(spikes), width=0.1, color=colors)
    # # ax.legend(
    # #     handles=proxy,
    # #     labels=label,
    # #     facecolor=ax.get_facecolor(),
    # #     edgecolor=None,
    # #     fancybox=False,
    # #     markerscale=3,
    # #     shadow=True)
    #
    # ax.set_title("Interspike Interval vs Time", fontweight='bold')
    # ax.set_xlabel('Time (s)')
    # ax.set_ylabel("ISI", fontweight='bold')
    # plt.tight_layout()
    # plt.show()

    histo = plot.Plot(spikes, events, colors)
    histo.histogram()


# x = my_list[0]
#
# bins = np.arange(x.iloc[0], x.iloc[-1], 0.1)
# y_height, _ = np.histogram(x, bins.reshape(-1))
#
# f, t, Sxx = signal.spectrogram(y_height)
# isi = np.diff(x)

# histo = plot.Plot(x, my_list[1], my_list[2])
# histo.histogram()
temp = []
