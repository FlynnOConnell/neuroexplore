from pathlib import Path
from data_utils import data_collection
import pandas as pd
from metricspace.py.distances import DistanceMatrix
from data_utils import io


def get_data(_dir):
    # default directory: r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf'
    dc = data_collection.DataCollection(_dir)
    dc.get_data(paradigm='sf', exclude=None)
    return dc


# data = load(r'C:\repos\neuroexplore\exclude\solidfood_raw.pkl')
# dists = []
# for filename, file in data.items():
#     print(file)
#     dists.append(DistanceMatrix(file))
# io.save(r'C:\repos\neuroexplore\exclude\solidfood_dists.pkl', dists)
x = 2
# load(r'C:\repos\neuroexplore\exclude\eating_dists.pkl')

