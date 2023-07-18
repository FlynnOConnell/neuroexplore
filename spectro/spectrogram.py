from __future__ import annotations
import numpy as np
from pathlib import Path
from nex.nexfile import Reader
import matlab
import matlab.engine
import scipy.io as sio


def to_numpy_objarray(list_of_lists):
    """
    Converts a list of lists into a numpy object array, where each element is a numpy array.

    Parameters:
        list_of_lists (list): A list of lists, where each inner list is intended to be a column vector.

    Returns:
        numpy.ndarray: A 1D numpy object array, where each element is a numpy array.
    """
    # Convert each sublist to a 2D numpy array with a single column
    array_of_arrays = [np.array(sublist) for sublist in list_of_lists]

    # Create an object array
    obj_arr = np.zeros((len(array_of_arrays),), dtype=object)

    # Assign each numpy array to the corresponding element in the object array
    for i in range(len(array_of_arrays)):
        obj_arr[i] = array_of_arrays[i]

    obj_arr = obj_arr.reshape(-1, 1)

    return obj_arr

tc_names = ["A", "AS", "MSG", "N", "Q", "S"]

files = []
for file_path in Path(r"").glob("*.nex"):
    reader = Reader(useNumpy=True)
    file = reader.ReadNexFile(str(file_path))
    files.append(file)

neurons = {}
data = {}
for i, nex in enumerate(files):
    # Get neurons and events timestamps
    for var in nex['Variables']:
        if var['Header']['Type'] == 0:
            data[var['Header']['Name']] = var['Timestamps']
        elif var['Header']['Type'] == 1 and var['Header']['Name'] in tc_names:
            data['events'][var['Header']['Name']] = var['Timestamps']

# Get a list of tuples of the form (stim, ts, spikes)
data_today_lst, data_yesterday_lst = [], []
for stim in data_today_dct['events']:
    for ts in data_today_dct['events'][stim]:
        data = np.array(data_today_dct['sig007a'])
        spks = np.where((data >= ts) & (data <= ts + 2))[0]
        adjusted_spks = data_today_dct['sig007a'][spks] - ts
        thisstim = (stim, ts, adjusted_spks.tolist())  # need to convert to list for matlab
        data_today_lst.append(thisstim)

for stim in data_yesterday_dct['events']:
    for ts in data_yesterday_dct['events'][stim]:
        data = np.array(data_yesterday_dct['sig007a'])
        spks = np.where((data >= ts) & (data <= (ts + 2)))[0]
        adjusted_spks = data_yesterday_dct['sig007a'][spks] - ts
        thisstim = (stim, ts, adjusted_spks.tolist())  # need to convert to list for matlab
        data_yesterday_lst.append(thisstim)

final_today = {
    'labels': [x[0] for x in data_today_lst],  # 'A', 'AS', 'MSG', 'N', 'Q', 'S'
    'cspks': [x[2] for x in data_today_lst],  # list of lists of spike times
}

final_yesterday = {
    'labels': [x[0] for x in data_yesterday_lst],  # 'A', 'AS', 'MSG', 'N', 'Q', 'S'
    'cspks': [x[2] for x in data_yesterday_lst],  # list of lists of spike times
}

cspks = to_numpy_objarray(final_today['cspks'] + final_yesterday['cspks'])
nsam = np.array([x[0].shape[0] for x in cspks])
qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))
Dists = pairwise_distances(np.array(cspks))

ANEAR = np.zeros((nsam.size, nsam.size, len(qvals)), dtype=np.float32)
for qv in range(len(qvals)):
    D_vec = Dists[:, :, qv]
    ANEAR[:, :, qv] = dc.distclust(D_vec, nsam)
