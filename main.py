import numpy as np
from scipy import stats, signal
import sys
import os

from nex import nexfile

# File to analyze
file = r'G:\PlexonData\Solid Food Data outputs\Cells recorded for both paradigms\SFN14 ' \
       r'2019-02-19 201902SF_OBEX.nex'

reader = nex.nexfile.Reader(useNumpy=True)
data = reader.ReadNexFile(file)

# indicate the names of the intervals which you want included in the heatmap
sf_labels = (
    'Spont',
    'Well (Empty)',
    'Well (Peanut)',
    'Well (Apply)',
    'Well (M. Chocolate)',
    'Eat (Apple)',
    'Eat (M. Chocolate)',
    'Eat (Apple)',

)

sf_events = (
    'Spont',
    'BLW_bout_Int',
    'FLW_bout_Int',
    'FRW_bout_Int',
    'BRW_bout_Int',
    'e_FR',
    'e_BR',
    'e_FL',
)

ts_adjust = .005  # time shift to adjust for the time it takes to lick
timestamps = {}

neurons = ()
trial_times = {}

timestamps['Start'] = np.asarray([0.0])
timestamps['Stop'] = np.asarray([0.0])

for event in data['Variables']:  # gets all the timestamps for each event
    # If the event is in the list to look for or the type is a neuron
    if event['Header']['Type'] == 0:
        # add the timestamps to the dict
        timestamps[event['Header']['Name']] = event['Timestamps']
        neurons = neurons + tuple([event['Header']['Name']])

for key, value in timestamps.items():
    if max(value) > timestamps['Stop'][0]:
        timestamps['Stop'] = np.asarray([max(value)])


