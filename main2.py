from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams, lines
import os
from nex import nexfile
from data_reader import ReadData
from data_formatter import DataFormatter
from helpers import funcs, ax_helpers

file = "SFN14_2018-12-07_vision_SF.nex"

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

data = ReadData(file, os.getcwd() + "/data/")
formatted = DataFormatter(data.raw_data, well_events, eating_events, colors)

data_trials = formatted.trials
data_nframes = formatted.neuroframes
# %%


# sf_events = well_events + eating_events + ['Spont']

binsize = .1  # binsize for binning spikes (seconds)
end = 6  # how long after interval start to display (Seconds)
start = -1  # how long before the interval start to display (seconds)


# %%
#
# rcParams.update(
#     {
#         "font.weight": "bold",
#         "font.family": "Arial",
#         'font.sans-serif': "Arial",
#         "axes.labelweight": "bold",
#         "xtick.major.width": "1.3",
#         "axes.facecolor": "w",
#         "axes.labelsize": 10,
#         "lines.linewidth": 1,
#     }
# )
#
# a_trial = trials['e_BR'][4]
# bins = np.arange(a_trial.Neuron.iloc[0], a_trial.Neuron.iloc[-1], binsize)
# # plt.hist(test.iloc[:, 0], histtype='step')
# y_height, x_bins = np.histogram(a_trial.Neuron, bins.reshape(-1))
# barcol = [funcs.get_matched_time(np.asarray(a_trial.Neuron), x)[0] for x in x_bins]
# a_ev = a_trial.Color.iloc[[np.where(a_trial.Neuron == x)[0][0] for x in barcol]]
#
# thisdict = {'Peanuts': 'orange',
#             'Well': 'blue',
#             'Movement': 'black'
#             }
#
# # %%
# fig, ax = plt.subplots()
#
# ax.bar(x_bins[:-1], y_height, width=binsize, color=a_ev)
# ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1)
# proxy, label = ax_helpers.get_handles_from_dict(thisdict, 3, marker="s")
# ax.legend(
#     handles=proxy,
#     labels=label,
#     facecolor=ax.get_facecolor(),
#     edgecolor=None,
#     fancybox=False,
#     markerscale=3,
#     shadow=True)
#
# ax.set_title("Frequency Histogram", fontweight='bold')
# ax.set_xlabel("Time (s)")
# ax.set_ylabel("Spikes / second", fontweight='bold')
#
# # ax.hlines(y=ax.get_ylim()[1], xmin=4, xmax=20, linewidth=2, color='r')
# plt.tight_layout()
# plt.show()
#
temp = []
