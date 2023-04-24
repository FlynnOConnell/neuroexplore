from pathlib import Path
from data_utils import data_collection


def output(files):
    from analysis import output
    outpath = r'C:\Users\Flynn\OneDrive\Desktop\temp\output.xlsx'
    return output.Output(outpath, files)


if __name__ == "__main__":

    datadir = Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles')
    data = data_collection.DataCollection(datadir)
    data.get_data(paradigm='sf')
    data.get_data(paradigm='rs')
    # data.get_stats()
    # files = data.files['sf']
    # data.output(r'C:\Users\Flynn\OneDrive\Desktop\temp\output.xlsx')
    # file = dh.Neuron(datadir / 'sf' / 'SFN11_2018-07-17_SF.nex')
    # collector = data_collection.DataCollection(datadir)
    # collector.get_data_sf()
    # sf_files = collector.files
    # out = output(sf_files)
    # out.concat_df()
    # mydf = out.file_outer
    # mydf.to_excel(r'C:\Users\Flynn\OneDrive\Desktop\temp\output.xlsx', index=False)
    # out.output(r'C:\Users\Flynn\OneDrive\Desktop\temp\output.xlsx')
    # Save the output DataFrame to an Excel file
    # output_df.T.to_excel(outpath, index=False)