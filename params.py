
class Params:
        def __init__(self, paradigm='food'):
            """Initialize parameters"""
            if not paradigm in ['food', 'stim']:
                raise ValueError("Paradigm must be 'food' or 'stim'")
 
            # eating params

            self.well_events = [
               'well_dchoc',
               'well_mchoc',
               'well_nuts',
               'well_apple',
                'well_brocc'
            ]
            self.eating_events = [
                 'eat_dchoc',
                 'eat_mchoc',
                 'eat_nuts',
                 'eat_apple',
                 'eat_brocc'
            ]
            self.order = ['Spont'] + self.well_events + self.eating_events
            self.colors = {key: "blue" for key in self.well_events}
            self.colors['e_FR'] = 'brown'
            self.colors['e_BR'] = 'orange'
            self.colors['e_FL'] = 'pink'
            self.colors['e_BL'] = 'red'
            self.colors['spont'] = 'gray'

            self.stats_columns = ['Spont'] + self.well_events + self.eating_events
            self.trial_stats_columns = ['animal',
                                        'date'] + self.stats_columns + self.stats_columns + self.stats_columns
            self.df_cols = ['animal', 'date'] + self.stats_columns + self.stats_columns
            self.opto_stats_columns = ['Spont', 'alleat', 'allwell', 'allzoneend'] + self.well_events + self.eating_events

            self.opto_events = ['alleat0_-2', 'allwell0_-2', 'allzoneend0_2',
                                'laser_eat0_2', 'laser_well0_2', 'laser_zoneprewell0_2',
                                'nolaser_eat0_2', 'nolaserwell0_2', 'nolaserzone_prewell0_2']


                #number of trials, mean time of event, sem of event

            self.tastants_opts = (
                        'T3_Citric_Acid',
                        'T3_ArtSal',
                        'T3_MSG',
                        'T3_NaCl',
                        'T3_Quinine',
                        'T3_Sucrose',
                        )

            # stim params
            self.binsize = 0.1
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
