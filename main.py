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

tstamps = formatted.timestamps['SPK08a']

average = [len(tstamps[np.where((interval[0] <= tstamps) & (tstamps <= interval[1]))]) / (interval[1] - interval[0]) for interval in formatted.interval_gen("Spont")]
np.mean(average)

# %%


# %%

temp = []
