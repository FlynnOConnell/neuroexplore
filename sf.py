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

data = load(r'C:\Users\Flynn\Dropbox\Lab\SF\data\save.pkl')

x = 4
