from __future__ import annotations
from data_utils import io
import numpy as np
from pathlib import Path
from nex.nexfile import Reader
# import metricspace as ms

qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))


# def run_spkd(spike_trains, qv, use_rust_backend=False):
#     if use_rust_backend:
#         neurodist = ms.spkd(spike_trains, qv, use_rs=use_rust_backend)
#     else:
#         neurodist = ms.spkd(spike_trains, qv, use_rs=use_rust_backend)
#     return neurodist


# def get_anear(pw_dists, n_samples):
#     nq = 28
#     anear_mat = np.zeros((len(n_samples), len(n_samples), nq))
#     for q in range(nq):
#         anear_mat[:, :, q] = ms.distclust(pw_dists[:, :, q], n_samples)
#     return anear_mat


# def get_info(conf_matrix):
#     nq = 28
#     h = np.zeros(nq)
#     mat_ja = np.zeros(nq)
#     mat_tp = np.zeros(nq)
#     for q in range(nq):
#         total = np.sum(conf_matrix[:, :, q])
#         mat_prob = conf_matrix[:, :, q] / total
#         h[q] = ms.tblxinfo(mat_prob)
#         mat_ja[q] = ms.tblxbi(conf_matrix[:, :, q], 'ja', None)
#         mat_tp[q] = ms.tblxbi(conf_matrix[:, :, q], 'tr', 0)
#     return h, mat_ja, mat_tp


def get_data(filepath):
    tc_events = ["T3_Citric_Acid", "T3_ArtSal", "T3_MSG", "T3_NaCl", "T3_Quinine", "T3_Sucrose"]
    neuraldata = []
    for file_path in Path(filepath).glob("*.nex"):
        reader = Reader(useNumpy=True)
        nexdata = reader.ReadNexFile(str(file_path))
        neuraldata.append(nexdata)

    data_today_dct = {'events': {}, }
    neurons = []

    for i, nex in enumerate(neuraldata):
        # Get neurons and events timestamps
        target = data_today_dct
        for var in nex['Variables']:
            if var['Header']['Type'] == 0:
                target[var['Header']['Name']] = var['Timestamps']
                neurons.append(var['Header']['Name'])
            elif var['Header']['Type'] == 1 and var['Header']['Name'] in tc_events:
                target['events'][var['Header']['Name']] = var['Timestamps']

    # Get a list of tuples of the form (stim, ts, spikes)
    data_today_lst = {}
    for neuron in neurons:
        thisneur = {}
        for stim in data_today_dct['events']:
            this_stim = []
            for ts in data_today_dct['events'][stim]:
                neurodata = np.array(data_today_dct[neuron])
                spks = np.where((neurodata >= ts) & (neurodata <= ts + 2))[0]
                adjusted_spks = data_today_dct[neuron][spks] - ts
                this_stim.append(adjusted_spks)
            thisneur[stim] = this_stim
        data_today_lst[neuron] = thisneur

    final_today = {
        'labels': [x[0] for x in data_today_lst],  # 'A', 'AS', 'MSG', 'N', 'Q', 'S'
        'cspks': [x[2] for x in data_today_lst],  # list of lists of spike times
    }

    _, counts = np.unique(final_today['labels'], return_counts=True)
    cspks_list = final_today['cspks']
    nsamples = np.array(counts)
    return cspks_list, nsamples


if __name__ == "__main__":
    datadir = Path.home() / "data" / "rs_test"
    cspks, nsam = get_data(datadir)
