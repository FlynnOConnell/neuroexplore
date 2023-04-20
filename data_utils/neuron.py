from pathlib import Path
import pandas as pd

from .neural_signals import NeuralSignals
from . import file_handling
from params import Params
from nex.nexfile import Reader

class Neuron:
    """
    Data class for holding data from a single neuron
    # TODO: Convert this to hold RF and SF data instances
    """
    file_full: Path

    def __init__(self, nexfilename: Path | str):
        self.errors = {}
        self.__scan(nexfilename)
        self.params = Params()
        self.__reader = Reader(useNumpy=True)
        if ".nex5" in nexfilename.suffix:
            self.__format(self.__reader.ReadNex5File(str(self.file_full)))
        elif ".nex" in nexfilename.suffix:
            self.__format(self.__reader.ReadNexFile(str(self.file_full)))
        else:
            raise Exception("File must be a .nex or .nex5 file")

    def __scan(self, nexfilename: Path | str):
        nexfilename = Path(nexfilename)
        if not Path(nexfilename).is_file():
            raise Exception(f"File {nexfilename} does not exist")
        if not ".nex" in nexfilename.suffix or ".nex5" in nexfilename.suffix:
            raise Exception("File must be a .nex or .nex5 file")
        self.file_full = nexfilename
        self.file_name = nexfilename.name
        self.__file_info = file_handling.parse_filename(self.file_name)

        self.animal = self.__file_info.animal
        self.date = self.__file_info.date
        self.paradigm = self.__file_info.paradigm
        if not self.paradigm in ['sf', 'SF']:
            raise Exception("File must be a solid food file ('sf' or 'SF')")

    def __repr__(self):
        return f'{self.animal} | {self.date} | {self.__alldata.neuron})'

    def __format(self, nexfile):
        self.__alldata = NeuralSignals(nexfile, self.params.well_events, self.params.eating_events, self.params.colors)
        self.__get_data()

    def __get_data(self):
        self.__alldata.add_trial_column()
        self.neurospikes = self.__alldata.neurostamps
        self.trials_dict = self.__alldata.trials
        self.neuron = self.__alldata.neuron
        self.df_sorted = self.__alldata.df_sorted
        self.df_binned = self.__alldata.df_binned
        self.df_filtered = self.__alldata.df_binned_filtered
        self.stats_trials = self.__alldata.stats_trials_df
        self.stats_file = self.__alldata.stats_file_df
        self.add_animal_date_neuron()
        self.stats_file = self.column_mapping(self.stats_file)
        self.stats_trials = self.column_mapping(self.stats_trials)

    @staticmethod
    def column_mapping(df: pd.DataFrame) -> pd.DataFrame:
        column_mapping = {
            'BLW_bout_Int': 'w_BL',
            'BRW_bout_Int': 'w_BR',
            'FLW_bout_Int': 'w_FL',
            'FRW_bout_Int': 'w_FR'
        }
        return df.rename(columns=column_mapping)

    def add_animal_date_neuron(self):
        self.stats_trials.insert(0, 'stat', self.stats_trials.index)
        self.stats_trials.reset_index(drop=True, inplace=True)
        self.stats_trials.insert(0, 'unit', self.neuron)
        self.stats_trials.insert(0, 'date', self.date)
        self.stats_trials.insert(0, 'animal', self.animal)
        self.stats_file.insert(0, 'stat', self.stats_file.index)
        self.stats_file.reset_index(drop=True, inplace=True)
        self.stats_file.insert(0, 'unit', self.neuron)
        self.stats_file.insert(0, 'date', self.date)
        self.stats_file.insert(0, 'animal', self.animal)
