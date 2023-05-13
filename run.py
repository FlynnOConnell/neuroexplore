from pathlib import Path
from data_utils import data_collection
import pandas as pd
import pickle
from data_utils import signals_stimuli as ss
from data_utils.file_handling import find_matching_files, parse_filename, get_nex
import os

def output(files):
    from analysis import output
    outpath = r'C:\Users\Flynn\OneDrive\Desktop\temp\output.xlsx'
    return output.Output(outpath, files)


if __name__ == "__main__":

    datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\rs')
    # data = data_collection.DataCollection(datadir)
    # # new_data = data_collection.DataCollection(datadir)
    # data = data.get_data(paradigm='replacements', functions_to_run=['spont'])

    files = find_matching_files(datadir, '*.nex*')
    allfile = []
    for file in files['rs']:
        try:
            session = parse_filename(file)
            nex = get_nex(os.path.join(datadir, file))
            data = ss.StimuliSignals(nex, file)
            data.run_stats(['spont'])
            allfile.append({session: data})
        except Exception as e:
            print(e)
            continue
    data1 = [file[session] for file in allfile for session in file.keys()]

    # final_five = pd.DataFrame()
    # final_fives = []
    # final_coh = pd.DataFrame()
    # final_cohs = []
    # final_lxl = pd.DataFrame()
    # final_lxls = []
    final_spont = pd.DataFrame()
    final_sponts = []
    # final_ili = pd.DataFrame()
    # final_ilis = []
    # final_bl = pd.DataFrame()
    # final_bls = []

    x = 0

    for stats in data1:
        # final_fives.append(stats.final_df['5L'])
        # final_cohs.append(stats.final_df['COH'])
        # final_lxls.append(stats.final_df['lxl'])
        final_sponts.append(stats.final_df['spont'])
        # final_ilis.append(stats.final_df['spont'])
        # final_bls.append(stats.final_df['ILI'])

    # df_fivelick = pd.concat([final_five] + final_fives, axis=0, ignore_index=True)
    # df_coh = pd.concat([final_coh] + final_cohs, axis=0, ignore_index=True)
    # df_lxl = pd.concat([final_lxl] + final_lxls, axis=0, ignore_index=True)
    df_spont = pd.concat([final_spont] + final_sponts, axis=0, ignore_index=True)
    # df_ili = pd.concat([final_ili] + final_ilis, axis=0, ignore_index=True)
    # df_bl = pd.concat([final_bl] + final_bls, axis=0, ignore_index=True)

    save = r'C:\Users\Flynn\OneDrive\Desktop\temp\rs_output_final4.xlsx'
    with pd.ExcelWriter(save) as writer:
        # df_fivelick.to_excel(writer, sheet_name='5L', index=False)
        # df_coh.to_excel(writer, sheet_name='COH', index=False)
        # df_lxl.to_excel(writer, sheet_name='lxl', index=False)
        df_spont.to_excel(writer, sheet_name='baseline', index=False)
        # df_ili.to_excel(writer, sheet_name='spont', index=False)
        # df_bl.to_excel(writer, sheet_name='ILI', index=False)
