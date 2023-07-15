from pathlib import Path
import numpy as np
import pandas as pd
from data_utils.file_handling import find_matching_files, parse_filename, get_nex
from data_utils import signals_food as sf
from data_utils.io import save

home = Path.home()
_dir = home / "data" / "sf"

alldata = {}
for file in _dir.glob('*.nex'):
    nexfile = get_nex(file)
    data = sf.EatingSignals(nexfile, str(file),)
    alldata[file.stem] = data
