from pathlib import Path

import numpy as np

from data_utils import data_collection
import pandas as pd
from data_utils import io
from data_utils.io import load


def get_data(_dir):
    # default directory: r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf'
    dc = data_collection.DataCollection(_dir)
    dc.get_data(paradigm='sf', exclude=None)
    return dc

data_two = get_data(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf')
files = data_two.files
alldata_sort = pd.DataFrame()

for file in files:

    filedata = pd.DataFrame(columns=['file', 'neuron', 'mean_spike_rate', 'std_spike_rate', 'sem_spike_rate'])
    animal1 = files[file]

    spont_sort = animal1.get_spont_intervs(sort=True)
    spont_ts_sort = animal1.get_spont_stamps(spont_sort)
    filename = None
    for neuron, df in spont_ts_sort.items():
        filename = [file for i in range(df.shape[0])]
    spont_ts_sort['file'] = filename
    alldata_sort = pd.concat([alldata_sort, spont_ts_sort], ignore_index=True)


with pd.ExcelWriter(r'C:\repos\neuroexplore\exclude\solidfood_normal.xlsx') as writer:
    alldata_sort.to_excel(writer, sheet_name='sorted')

# dists = []
# for filename, file in data.items():
#     print(file)
#     dists.append(DistanceMatrix(file))
x = 2
# load(r'C:\repos\neuroexplore\exclude\eating_dists.pkl')

