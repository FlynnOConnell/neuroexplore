
class Params:
        def __init__(self):
            """Initialize parameters"""
            self.filename = 'SFN16_2019-03-25_SF.nex'
            self.directory = r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf'
            self.binsize = 0.1
            self.well_events = [
               'BLW_bout_Int',
               'BRW_bout_Int',
               'FLW_bout_Int',
               'FRW_bout_Int']
            self.eating_events = [
                 'e_BL',
                 'e_BR',
                 'e_FL',
                 'e_FR'
            ]

            self.order = ['Spont'] + self.well_events + self.eating_events
            self.mapping = {
                'BLW_bout_Int': 'w_BL',
                'BRW_bout_Int': 'w_BR',
                'FLW_bout_Int': 'w_FL',
                'FRW_bout_Int': 'w_FR'
            }

            self.colors = {key: "blue" for key in self.well_events}
            self.colors['e_FR'] = 'brown'
            self.colors['e_BR'] = 'orange'
            self.colors['e_FL'] = 'pink'
            self.colors['e_BL'] = 'red'
            self.colors['spont'] = 'gray'

            self.stats_columns = ['Spont'] + self.well_events + self.eating_events
            self.df_cols = ['animal', 'date'] + self.stats_columns + self.stats_columns

            self.tastants_opts = (
                        'T2_Citric_Acid',
                        'T2_ArtSal',
                        'T2_MSG',
                        'T2_NaCl',
                        'T2_Quinine',
                        'T2_Sucrose',
                        'Citric_Acid',
                        'ArtSal',
                        'MSG',
                        'NaCl',
                        'Quinine',
                        'Sucrose',
                        )






