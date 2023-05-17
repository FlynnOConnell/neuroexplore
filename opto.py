from pathlib import Path
from data_utils import data_collection

datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf')
outpath = Path(r'C:\Users\Flynn\OneDrive\Desktop\temp\evstats4.xlsx')

alldata = data_collection.DataCollection(datadir, opto=False)
alldata.get_data()
alldata.get_session_stats()
alldata.output(alldata.session_stats, outpath, sheet_name='session_stats')


if __name__ == '__main__':
    pass
