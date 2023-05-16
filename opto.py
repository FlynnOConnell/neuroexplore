
from pathlib import Path
import pandas as pd
import numpy as np
from nex.nexfile import Reader
from params import Params
from data_utils import signals_food as sf
from data_utils.file_handling import find_matching_files, parse_filename, get_nex
from data_utils import data_collection, file_handling, signals_food
import xlsxwriter

name_mapping = {
    'grooming': 'Grooming',
    'Grooming (1)': 'Grooming',

    'w_apple': 'well_apple',
    'w_a': 'well_apple',
    'e_a': 'eat_apple',
    'e_apple': 'eat_apple',

    'w_nuts': 'well_nuts',
    'w_n': 'well_nuts',
    'e_nuts': 'eat_nuts',
    'e_n': 'eat_nuts',

    'w_c': 'well_choc',
    'e_c': 'eat_choc',
    'w_choc': 'well_choc',
    'e_choc': 'eat_choc',

    'w_banana': 'well_banana',
    'e_banana': 'eat_banana',
    'w_b': 'well_broccoli',
    'e_b': 'eat_broccoli',
    'w_broccoli': 'well_broccoli',
    'e_broccoli': 'eat_broccoli',
}

bad = ['eating_unknown', 'tail', 'Tail', 'e_unknown', 'Interbout_Int', 'nodata']


def output(df, outpath):
    writer = pd.ExcelWriter(outpath, engine='xlsxwriter')
    df.stats.to_excel(writer, sheet_name='stats', index=False)
    writer.save()

def filter_events(df):
    df['column_name'] = df['column_name'].replace(name_mapping)

datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf')
outpath = Path(r'C:\Users\Flynn\OneDrive\Desktop\temp\evstats4.xlsx')

session_info = parse_filename(datadir.name)

alldata = data_collection.DataCollection(datadir, opto=False)
alldata.get_data()
alldata.get_eating_stats()
df = alldata.stats

df = df[~df.event.isin(bad)]
df['event'] = df['event'].replace(name_mapping)
cols = df.columns.tolist()
cols = cols[-2:] + cols[:-2]
df = df[cols]

# writer = pd.ExcelWriter(outpath, engine='xlsxwriter')
# df.to_excel(writer, index=False)
# writer.save()
# workbook  = writer.book
# worksheet = writer.sheets['Sheet1']
#
# # Define a range for the color formatting
# color_range = "A2:A{}".format(len(df) + 1)
#
# # Create border formats
# border_format = workbook.add_format({'bottom':5})
# thin_border_format = workbook.add_format({'top': 1, 'top_color': 'black'})
#
# prev_val = df.iloc[0]['animal']
# for index, value in enumerate(df['animal']):
#     if value != prev_val:
#         worksheet.conditional_format(index, 0, index, df.columns.size, {'type': 'formula', 'criteria': 'True', 'format': border_format})
#     prev_val = value
#
# prev_val = df.iloc[0]['neuron']
# for index, value in enumerate(df['neuron']):
#     if value != prev_val:
#         worksheet.conditional_format(index + 1, 2, index + 1, df.columns.size, {'type': 'formula', 'criteria': 'True', 'format': thin_border_format})
#     prev_val = value
# writer.save()
