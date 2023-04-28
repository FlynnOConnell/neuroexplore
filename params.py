
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
                        'T3_Citric_Acid',
                        'T3_ArtSal',
                        'T3_MSG',
                        'T3_NaCl',
                        'T3_Quinine',
                        'T3_Sucrose',
                        )

            self.trial_end = 4
            self.base = 2
            self.bin5L = 0.1
            self.lxl_method = "CHISQ"
            # number of samples for each monte carlo approximation (exact multinomial test)
            self.monte_n = 100000
            # Maximum p-value (percentile) to be considered significant,
            self.chi_alpha = 0.05
            # Length of post-stimulus time to consider
            self.lick_max = .150
            self.lick_window = .015
            self.wil_alpha = .05






