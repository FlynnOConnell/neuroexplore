from pathlib import Path
from data_utils import data_collection
import pandas as pd
import pickle

def load(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def save(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)
        print('saved')

sessions = load(r'C:\Users\Flynn\OneDrive\Desktop\temp\save.pkl')

all_five = []
all_lxl = []

for session in sessions:
    for ix, signal in session.items():
        fl_session = pd.concat([signal.fl_baseline, signal.fl_adjmags, signal.fl_duration, signal.fl_latency, signal.fl_duration, signal.fl_trials], axis=1)
        lxl_session = pd.concat([signal.lxl_responses, signal.lxl_trials], axis=1)

        all_five.append(fl_session)
        all_lxl.append(lxl_session)

five_df = pd.concat(all_five, axis=0, ignore_index=False)
lxl_df = pd.concat(all_lxl, axis=0, ignore_index=False)
savename = Path(r'C:\Users\Flynn\OneDrive\Desktop\temp\save.xlsx')

with pd.ExcelWriter(savename, engine='xlsxwriter', ) as writer:
    five_df.to_excel(writer, 'five_lick', float_format='%.3f', header=True, index=True)
    lxl_df.to_excel(writer, 'lxl', float_format='%.3f', header=True, index=True)

if __name__ == "__main__":
    pass
    # datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\rs')
    # data = data_collection.DataCollection(datadir)
    # data.get_data(paradigm='rs',  functions_to_run=['changepoint', 'chisquare'])
