from __future__ import annotations
from helpers.funcs import interval
import numpy as np
from pathlib import Path
from nex.nexfile import Reader
import metricspace as ms
from data_utils.io import load, save

qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))
