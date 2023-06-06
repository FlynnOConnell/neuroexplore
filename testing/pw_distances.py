from __future__ import annotations

import metricspace
import numpy as np
from pathlib import Path
from nex.nexfile import Reader
from data_utils import io
import time
from metricspace import *
import os
import rs_distances.rs_distances

# Set the RUST_BACKTRACE environment variable to '1'
os.environ["RUST_BACKTRACE"] = "1"
qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))


def get_anear(pw_dists, n_samples):
    nq = 28
    anear_mat = np.zeros((len(n_samples), len(n_samples), nq))
    for q in range(nq):
        anear_mat[:, :, q] = distclust(pw_dists[:, :, q], n_samples)
    return anear_mat


def get_info(conf_matrix):
    nq = 28
    h = np.zeros(nq)
    mat_ja = np.zeros(nq)
    mat_tp = np.zeros(nq)
    for q in range(nq):
        total = np.sum(conf_matrix[:, :, q])
        mat_prob = conf_matrix[:, :, q] / total
        h[q] = tblxinfo(mat_prob)
        mat_ja[q] = tblxbi(conf_matrix[:, :, q], 'ja', None)
        mat_tp[q] = tblxbi(conf_matrix[:, :, q], 'tr', 0)

    return h, mat_ja, mat_tp


def get_data():
    tc_events = ["T1_A1", "T1_S1"]
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
            neurodata = np.array(data_today_dct['sig2_C1'])
            spks = np.where((neurodata >= ts) & (neurodata <= ts + 3))[0]
            adjusted_spks = data_today_dct['sig2_C1'][spks] - ts
            thisstim = (stim, ts, adjusted_spks)  # need to convert to list for matlab
            data_today_lst.append(thisstim)

    final_today = {
        'labels': [x[0] for x in data_today_lst],  # 'A', 'AS', 'MSG', 'N', 'Q', 'S'
        'cspks': [x[2] for x in data_today_lst],  # list of lists of spike times
    }
    cspks = final_today['cspks']
    # neurodist = metricspace.spkd(cspks, qvals)
    # start = time.time()
    neurodist = rs_distances.calculate_spkd_impl(cspks, qvals,)

    # end = time.time()
    # print(end - start)
    _, counts = np.unique(final_today['labels'], return_counts=True)
    nsamples = np.array(counts)
    # today = {'dists': neurodist, 'nsam': nsamples}
    return neurodist, nsamples


data, nsam = get_data()
# to_load = 1
# data = get_data() if to_load else io.load(r'C:\Users\Flynn\Desktop\data\dists.pkl')
# dists = data['dists']
# nsam = data['nsam']
# to_save = {'dists': data, 'nsam': [20, 19]}
# io.save(r'C:\Users\Flynn\Desktop\data\spk_plus_five.pkl', to_save)

# dists = data['dists']
# nsam = data['nsam']
# anear = get_anear(dists, nsam)
# info, ja, tp = get_info(anear)
#
stopper = 2
