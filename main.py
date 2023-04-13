import signal
from collections import defaultdict, OrderedDict
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import rcParams, lines
from scipy import signal
from scipy.ndimage import gaussian_filter
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
colors['spont'] = 'gray'
x_eating = ['brown', 'orange', 'pink', 'red']
Events_to_number = {
    "Spont": 0,
    "Well": 1,
    "Eating": 2,
    "Transition": np.nan
}

savepath = "/Users/flynnoconnell/Pictures/sf"

_data = ReadData(file, os.getcwd() + "/data/")
data = DataFormatter(_data.raw_data, well_events, eating_events, colors)
data_trials = data.trials

aa = data.trials['e_BR'][1]

# ix = 0
# for index, info in aa.iterrows():
#     ix = 1
#     if ix == 1:
#         pre_well_idx = index
#         ix = 2
#
#     if (info["EventNum"] == 1) & (ix == 2):
#         wellstart_idx = index
#         ix = 1
#     if (info["EventNum"] == 1) & (aa.iloc[index + 1, 3] != 1):
#         wellend_idx = index
#         ix = 1


def single_plot(plot_data):
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

def single_histo(frame, sigma=1, width=0.1):
    fig, ax = plt.subplots()
    ax.bar(frame.neuron,gaussian_filter(frame.counts, sigma=sigma), width=width, color=frame.color)
    ax.set_title("Single Trial Histogram", fontweight='bold')
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (hz)", fontweight='bold')
    ax.set_frame_on(False)
    plt.tight_layout()
    plt.show()



# z_df = None
# z_time = None
# z_length = None
# i = 0
# for x, arr, length in data.generate_spectrogram_data():
#     i += 1
#     if i == 2:
#         z_df = x
#         z_time = arr
#         z_length = length
#
# trial = data_trials["e_BR"][1]
#
# trial.loc[:,'counts'] = trial.groupby(['bins'])['color'].transform('count')
#
for event, trial in data_trials.items():
    if trial:
        for trial_num, this_trial_df in data_trials[event].items():
        # data_trials[event][trial]
            single_histo(this_trial_df, 0)

df1 = data_trials['e_BR'][1]
df_e = df1[df1['event'].isin(eating_events)]
df_w = df1[(df1.index <= df_e.index[0]) & (df1['event'].isin(well_events))]
df_prew = df1[(df1.index <= df_w.index[0])]

uni = data.df_sorted.loc[data.df_sorted.mid.drop_duplicates().index, ['color', 'counts', 'mid']]

df_spont = uni[uni['color'] == 'gray']
df_all_e = uni[uni['color'].isin(x_eating)]
df_all_w = uni[uni['color'] == 'blue']

df_all_ev = pd.concat([df_spont, df_all_e, df_all_w], axis=0)
df_all_ev = df_all_ev.drop('mid', axis=1).reset_index(drop=True)
df_all_ev = df_all_ev.reset_index(drop=True)

fig, ax = plt.subplots()
ax.bar(df_all_ev.index, gaussian_filter(df_all_ev.counts, sigma=2), width=0.1, color=df_all_ev.color)
ax.set_title("Everything", fontweight='bold')
ax.axes.get_xaxis().set_visible(False)
ax.set_ylabel("Frequency (hz)", fontweight='bold')
ax.set_frame_on(False)
# plt.tight_layout()
plt.savefig(r'C:\Users\Flynn\Dropbox\Lab\fig')


# ax = sns.barplot(x=trial['Bins'], y=trial['Neuron'])
# ax.tick_params(axis='x', labelrotation=90)
# plt.show()

# eatingdata = y[y[:,idx]]

# frequency_range =  np.round(np.linspace(0, np.amax(y), 15), 0)
# time =  np.round(np.arange(-2, 5, 0.1)[:-1], 1)
#
# spectrogram = np.zeros(shape=(int(np.amax(y)-1), time.shape[0]))
#
# for i in range(y.shape[1]):
#     col = y[:, i]
#     spectrogram[:, i], _ = np.histogram(col, range(0, int(np.amax(y)), 1))
#
# # spectrogram = gaussian_filter(spectrogram, sigma=2)
# spec = spectrogram[::-1]
# fix, ax = plt.subplots()
# ax = sns.heatmap(gaussian_filter(y, sigma=2), cmap='plasma', ax=ax, cbar_kws={'label': 'Spikes/s'})
#

# xlab = [-2, 0, 1, 2, 3, 4, 5, 6]
# xpos = [0, 10, 20, 30, 40, 50, 60, 70]
# ypos = [10, 0]
# ylab = [0, 150]

# ax.xaxis.set_ticks([0, 10, 20, 30, 40, 50])
# ax.xaxis.set_ticklabels([-2, 0, 1, 2, 3, 4])
# ax.yaxis.set_ticks(ypos)
# ax.yaxis.set_ticklabels(ylab)




