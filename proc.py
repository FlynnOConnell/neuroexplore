from pathlib import Path
from helpers.heatmap import sf_heatmap_stretch
from data_utils.file_handling import find_matching_files, parse_filename, get_nex
from data_utils.signal import Signals
x = 0
save_dir = Path.home() / 'data' / 'res' / 'heatmaps' / 'cutoff'
search_dir = Path.home() / 'data' / 'sf'
files_not_run = []
for file in search_dir.glob('*SFN11_2018-08-16*'):
    nexfile = get_nex(file)
    data = Signals(nexfile, file.name)
    for neuron in data.test_df:
        if data.test_df[neuron].empty:
            files_not_run.append(f'{file}_{neuron}')
            continue
        filename = f'{save_dir}/{file.stem}_{neuron}'
        sf_heatmap_stretch(data.test_df[neuron], spont_ts=data.spont[neuron], cutoff=6,
                           cmap='seismic', sigma=0, filename=filename, save=True
                           )
