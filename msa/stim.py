from pathlib import Path
from data_utils import data_collection
import pandas as pd
import pickle
from data_utils import io
from data_utils.io import load
from data_utils.file_handling import get_nex

datadir = r'C:\Users\Flynn\Dropbox\SF2\SFN07_2018-05-04_RS_OBEX.nex'
dc = data_collection.DataCollection(r'C:\Users\Flynn\Dropbox\Sf2')
dc.get_data()
nexfile = get_nex(datadir)