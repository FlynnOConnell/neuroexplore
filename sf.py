from pathlib import Path
from data_utils import data_collection
import pandas as pd
import pickle
from metricspace.py.distances import DistanceMatrix


def get_data(_dir):
    # default directory: r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf'
    dc = data_collection.DataCollection(_dir)
    dc.get_data(paradigm='sf', exclude=None)
    return dc

def load(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def save(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)
        print('saved')

def save_sessions(data=None):
    if not data:
        data = load(r'C:\Users\Flynn\OneDrive\Desktop\temp\save.pkl')

    all_five = []
    all_lxl = []

    for filename, signal in data.items():
        fl_session = pd.concat(
            [signal.fl_baseline, signal.fl_adjmags, signal.fl_duration, signal.fl_latency, signal.fl_duration,
             signal.fl_trials], axis=1)
        lxl_session = pd.concat([signal.lxl_responses, signal.lxl_trials], axis=1)
        fl_session['animal'] = filename
        lxl_session['animal'] = filename

    five_df = pd.concat(all_five, axis=0, ignore_index=False)
    lxl_df = pd.concat(all_lxl, axis=0, ignore_index=False)
    savename = Path(r'C:\Users\Flynn\OneDrive\Desktop\temp\savelast.xlsx')

    with pd.ExcelWriter(savename, engine='xlsxwriter', ) as writer:
        five_df.to_excel(writer, 'five_lick', float_format='%.3f', header=True, index=True)
        lxl_df.to_excel(writer, 'lxl', float_format='%.3f', header=True, index=True)

load(r'C:\repos\neuroexplore\exclude\eating_dists.pkl')


x=2

# for qv in range(len(qvals)):
#      D_vec = Dists[:, :, qv]
#      ANEAR[:, :, qv] = dc.distclust(D_vec, nsam)
