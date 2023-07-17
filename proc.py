from pathlib import Path
from helpers.heatmap import sf_heatmap, sf_heatmap_stretch
from data_utils.file_handling import find_matching_files, parse_filename, get_nex
from data_utils.signal import Signals

data_dir = Path.home() / 'data' / 'sf' / 'SFN11_2018-08-16_SF.nex'
nex_file = get_nex(data_dir)
data = Signals(nex_file, "SFN11_2018-08-16_SF.nex")
test_df = data.test_df
# for neuron in test_df:
#     sf_heatmap_stretch(test_df[neuron], cmap='plasma', sigma=0)
for neuron in test_df:
    sf_heatmap_stretch(test_df[neuron], cmap='plasma', sigma=0)
