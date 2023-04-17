import os
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.fft import fft, ifft
from data_reader import ReadData
from data_formatter import DataFormatter
from params import Params
from matplotlib import rcParams
from helpers import plot

rcParams.update(
    {
        "font.weight": "bold",
        "font.family": "Arial",
        'font.sans-serif': "Arial",
        "axes.labelweight": "bold",
        "xtick.major.width": "1.3",
        "axes.facecolor": "w",
        "axes.labelsize": 10,
        "lines.linewidth": 1,
    }
)

class Data:
    def __init__(self):
        self.data = {}
        self.params = Params()
        self.nexdata = ReadData(self.params.filename, os.getcwd() + self.params.directory)
        self.alldata = DataFormatter(self.nexdata.raw_data, self.params.well_events, self.params.eating_events, self.params.colors)
        self.get_data()

    def get_data(self):
        self.data['nexdata'] = self.alldata
        self.data['params'] = self.params
        self.data['timestamps'] = self.alldata.neurostamps
        self.data['trials'] = self.alldata.trials



