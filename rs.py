from pathlib import Path
from data_utils import data_collection
import pandas as pd
import pickle


def load(file):
    with open(file, "rb") as f:
        return pickle.load(f, fix_imports=True)


def save(file, save_data):
    with open(file, "wb") as f:
        pickle.dump(save_data, f)
        print('saved')


def save_sessions(session_data=None):
    if not session_data:
        session_data = load(r'C:\Users\Flynn\OneDrive\Desktop\temp\save.pkl')

    all_five = []
    all_lxl = []

    for filename, signal in session_data.items():
        fl_session = pd.concat(
            [signal.fl_baseline, signal.fl_adjmags, signal.fl_duration, signal.fl_latency, signal.fl_duration,
             signal.fl_trials], axis=1)
        lxl_session = pd.concat([signal.lxl_responses, signal.lxl_trials], axis=1)
        fl_session['animal'] = filename
        lxl_session['animal'] = filename
        all_five.append(fl_session)
        all_lxl.append(lxl_session)

    five_df = pd.concat(all_five, axis=0, ignore_index=False)
    lxl_df = pd.concat(all_lxl, axis=0, ignore_index=False)
    savename = Path(r'C:\Users\Flynn\OneDrive\Desktop\temp\savelast.xlsx')

    with pd.ExcelWriter(savename, engine='xlsxwriter', ) as writer:
        five_df.to_excel(writer, 'five_lick', float_format='%.3f', header=True, index=True)
        lxl_df.to_excel(writer, 'lxl', float_format='%.3f', header=True, index=True)


if __name__ == "__main__":
    files = load(r"C:\Users\Flynn\Dropbox\Lab\SF\data\save.pkl")
