from __future__ import annotations
import numpy as np
from pathlib import Path
from nex.nexfile import Reader
from metricspace.py.model import distclust, pw_distances
from data_utils import io
import time


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


def get_data(cspks_only=False):
    tc_events = ["T1_A1", "T1_S1", "T1_Q1", "T1_N1", "T1_M1"]
    # tc_events = ["T3_Citric_Acid", "T3_ArtSal", "T3_MSG", "T3_NaCl", "T3_Quinine", "T3_Sucrose"];

    files = []
    for file_path in Path(r'C:\Users\Flynn\Dropbox\Lab\SF\nexfiles\rs\test').glob("*.nex"):
        reader = Reader(useNumpy=True)
        file = reader.ReadNexFile(str(file_path))
        files.append(file)

    neurons = {}
    data_today_dct = {'events': {}, }

    for i, nex in enumerate(files):
        # Get neurons and events timestamps
        target = data_today_dct
        for var in nex['Variables']:
            if var['Header']['Type'] == 0:
                target[var['Header']['Name']] = var['Timestamps']
            elif var['Header']['Type'] == 1 and var['Header']['Name'] in tc_events:
                target['events'][var['Header']['Name']] = var['Timestamps']

    # Get a list of tuples of the form (stim, ts, spikes)
    data_today_lst = []
    for stim in data_today_dct['events']:
        for ts in data_today_dct['events'][stim]:
            data = np.array(data_today_dct['sig2_C1'])
            spks = np.where((data >= ts) & (data <= ts + 2))[0]
            adjusted_spks = data_today_dct['sig2_C1'][spks] - ts
            thisstim = (stim, ts, adjusted_spks)  # need to convert to list for matlab
            data_today_lst.append(thisstim)

    final_today = {
        'labels': [x[0] for x in data_today_lst],  # 'A', 'AS', 'MSG', 'N', 'Q', 'S'
        'cspks': [x[2] for x in data_today_lst],  # list of lists of spike times
    }
    cspks = final_today['cspks']

    if cspks_only:
        return cspks
    else:
        dists = pw_distances.spkd_pw(cspks, qvals)
        _, counts = np.unique(final_today['labels'], return_counts=True)
        nsam = np.array(counts)
        data = {'dists': dists, 'nsam': nsam}
        return data

to_load = 1
qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))
data = get_data() if to_load else io.load(r'C:\Users\Flynn\Desktop\data\dists.pkl')
dists = data['dists']
nsam = data['nsam']
# io.save(r'C:\Users\Flynn\Desktop\data\dists.pkl', data)
anear = np.zeros((len(data['nsam']), len(data['nsam']), len(qvals)))
for nq in range(0, len(qvals)):
    a_vec = dists[:, :, nq]
    anear[:, :, nq] = distclust(a_vec, nsam)
data['anear'] = anear
io.save(r'C:\Users\Flynn\Desktop\data\dists_normal.pkl', data)


x = 5
