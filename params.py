
class Params:
        def __init__(self):
            """Initialize parameters"""
            self.filename = 'SFN16_2019-03-25_SF.nex'
            self.directory = r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\sf'

            self.well_events = ['BLW_bout_Int',
               'FLW_bout_Int',
               'FRW_bout_Int',
               'BRW_bout_Int']
            self.eating_events = ['e_FR',
                 'e_BR',
                 'e_FL',
                 'e_BL']
            self.colors = {key: "blue" for key in self.well_events}
            self.colors['e_FR'] = 'brown'
            self.colors['e_BR'] = 'orange'
            self.colors['e_FL'] = 'pink'
            self.colors['e_BL'] = 'red'
            self.colors['spont'] = 'gray'




