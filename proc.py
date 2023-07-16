from pathlib import Path
import numpy as np
import pandas as pd
from data_utils.file_handling import find_matching_files, parse_filename, get_nex
from data_utils import signals_food as sf
from data_utils.signals_food import get_spike_times_within_interval
from data_utils.io import save, load

home = Path.home()
data = home / 'data' / 'res' / 'sf' / 'sf_data.pkl'
data = load(data)

testfile = data['SFN07_2018-05-04_SF']
testfile.event_df.sort_values(by='start_time', inplace=True)


x = 4