from pathlib import Path
from data_utils import data_collection
from data_utils.file_handling import parse_filename
import pandas as pd
import numpy as np
import pickle
from params import Params

def load(file):
    with open(file, "rb") as f:
        return pickle.load(f)

def save(file, data):
    with open(file, "wb") as f:
        pickle.dump(data, f)
        print('saved')


def concat_dataframe_to_dataframe(df1, df2):
    """
    Concatenates two pandas DataFrames along the rows, adding NaN for any missing columns in either DataFrame.

    The function first finds and adds any new columns that are present in `df2` but not in `df1`. It then finds
    and adds any missing columns that are present in `df1` but not in `df2`. The dataframes are then concatenated
    along the rows (axis=0). The resulting DataFrame has the same column order as `df1` and the index is reset.

    Args:
        df1 (pandas.DataFrame): The first DataFrame.
        df2 (pandas.DataFrame): The second DataFrame to be concatenated to the first DataFrame.

    Returns:
        pandas.DataFrame: The resulting concatenated DataFrame.

    Examples:
        >>> df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
        >>> df2 = pd.DataFrame({'B': [5, 6], 'C': [7, 8]})
        >>> concat_dataframe_to_dataframe(df1, df2)
           A  B   C
        0  1  3 NaN
        1  2  4 NaN
        2 NaN  5 7.0
        3 NaN  6 8.0
    """
    # Find the columns that are in df2 but not in df1
    new_columns = df2.columns.difference(df1.columns)

    # Add the new columns to df1 with NaN values
    for col in new_columns:
        df1[col] = np.nan

    # Find the columns that are in df1 but not in df2
    missing_columns = df1.columns.difference(df2.columns)

    # Add the missing columns to df2 with NaN values
    for col in missing_columns:
        df2[col] = np.nan

    updated_df = pd.concat([df1, df2], axis=0)

    # Reorder the columns based on the original df1
    updated_df = updated_df[df1.columns]
    return updated_df

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
        all_five.append(fl_session)
        all_lxl.append(lxl_session)

    five_df = pd.concat(all_five, axis=0, ignore_index=False)
    lxl_df = pd.concat(all_lxl, axis=0, ignore_index=False)
    savename = Path(r'C:\Users\Flynn\OneDrive\Desktop\temp\savelast.xlsx')

    with pd.ExcelWriter(savename, engine='xlsxwriter', ) as writer:
        five_df.to_excel(writer, 'five_lick', float_format='%.3f', header=True, index=True)
        lxl_df.to_excel(writer, 'lxl', float_format='%.3f', header=True, index=True)

datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf')

dc = data_collection.DataCollection(datadir)
dc.get_data(paradigm='sf', exclude=None)

dfs = []
resp = pd.DataFrame()
for name, file in dc.files.items():
    sig = file.spont_stats
    sig.insert(0, 'animal', name)
    dfs.append(sig)

resp = pd.concat(dfs, axis=0, ignore_index=True)

dc.get_session_stats()
params = Params()
info_df = pd.DataFrame(columns=['animal', 'date'])
means = pd.DataFrame(columns=params.stats_columns)
sems = pd.DataFrame(columns=params.stats_columns)
std = pd.DataFrame(columns=params.stats_columns)
for filename, data in dc.files.items():
    session_info = parse_filename(filename)
    session = [session_info.animal, session_info.date]
    for neuron in data.neurons.keys():
        # one row per neuron, fill timestamps
        info_df.loc[len(info_df)] = session
    means = concat_dataframe_to_dataframe(means, data.means)
    sems = concat_dataframe_to_dataframe(sems, data.sems)
    std = concat_dataframe_to_dataframe(std, data.stds)

std.insert(0, 'data', info_df['date'].values)
std.insert(0, 'animal', info_df['animal'].values)

dc.output(resp, r"C:\Users\Flynn\OneDrive\Desktop\temp\std.xlsx")

# def format_xl(writer, sheets, df):
#     workbook = writer.book
#     # border formats
#     border_format = workbook.add_format({'bottom': 5})
#     thin_border_format = workbook.add_format({'top': 1, 'top_color': 'black'})
#
#     if not isinstance(sheets, list):
#         sheets = [sheets]
#
#     for sheet in sheets:
#         worksheet = writer.sheets['Sheet1']
#         prev_val = df.iloc[0]['animal']
#         for index, value in enumerate(df['animal']):
#             if value != prev_val:
#                 worksheet.conditional_format(index, 0, index, df.columns.size,
#                                              {'type': 'formula', 'criteria': 'True', 'format': border_format})
#             prev_val = value
#
#         prev_val = df.iloc[0]['neuron']
#         for index, value in enumerate(df['neuron']):
#             if value != prev_val:
#                 worksheet.conditional_format(index + 1, 2, index + 1, df.columns.size,
#                                              {'type': 'formula', 'criteria': 'True', 'format': thin_border_format})
#             prev_val = value
#     writer.save()

# writer = pd.ExcelWriter(r'C:\Users\Flynn\OneDrive\Desktop\temp\std.xlsx', engine='xlsxwriter')
#
# format_xl(writer, ['Sheet1'], std)

x=5
if __name__ == "__main__":
    pass
