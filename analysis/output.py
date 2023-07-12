import pandas as pd
from pathlib import Path
from params import Params

params = Params()


class Output:
    """Class for outputting a single Signals data object to excel files"""

    def __init__(self, data: pd.DataFrame, save_directory, ):
        self.data = data
        self.directory = Path(save_directory)
        assert isinstance(self.data, pd.DataFrame), 'Data must be a pandas DataFrame'

    def save_file(self, filename, sheetname, ):
        """
            Output the data to an Excel file.
            If sheetname is None, the data will be saved to a new sheet to be named by the engine.
         """
        if not self.directory.exists():
            self.directory.mkdir(parents=True)
        savename = self.directory / f'{filename}.xlsx'
        with pd.ExcelWriter(savename, engine='xlsxwriter', ) as writer:
            self.data.to_excel(writer, sheetname, float_format='%.3f', header=True, index=True)
            print(f'Saved {savename}')

    @staticmethod
    def format(self, writer, sheets, df):

        workbook = writer.book
        # border formats
        border_format = workbook.add_format({'bottom': 5})
        thin_border_format = workbook.add_format({'top': 1, 'top_color': 'black'})

        if not isinstance(sheets, list):
            sheets = [sheets]

        for sheet in sheets:
            worksheet = writer.sheets['Sheet1']
            prev_val = df.iloc[0]['animal']
            for index, value in enumerate(df['animal']):
                if value != prev_val:
                    worksheet.conditional_format(index, 0, index, df.columns.size,
                                                 {'type': 'formula', 'criteria': 'True', 'format': border_format})
                prev_val = value

            prev_val = df.iloc[0]['neuron']
            for index, value in enumerate(df['neuron']):
                if value != prev_val:
                    worksheet.conditional_format(index + 1, 2, index + 1, df.columns.size,
                                                 {'type': 'formula', 'criteria': 'True', 'format': thin_border_format})
                prev_val = value
        writer.save()
