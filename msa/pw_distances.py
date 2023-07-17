from __future__ import annotations
from helpers.funcs import interval
import numpy as np
from pathlib import Path
from nex.nexfile import Reader
import metricspace as ms
from data_utils.data_io import load, save

qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))


def get_data(fullfile, t=2):
    tc_events = ["T3_Citric_Acid", "T3_MSG", "T3_NaCl", "T3_Quinine", "T3_Sucrose"]
    reader = Reader(useNumpy=True)
    nexdata = reader.ReadNexFile(str(fullfile))

    neurons = {}
    events = {}
    for var in nexdata['Variables']:
        if var['Header']['Type'] == 0:
            neurons[var['Header']['Name']] = var['Timestamps']
            events[var['Header']['Name']] = var['Timestamps']
        elif var['Header']['Type'] == 1 and var['Header']['Name'] in tc_events:
            pass

    evs = {ev: [] for ev in events.keys()}
    for event, tstamps in events.items():
        input_spks = tstamps
        interv = interval(input_spks, 5)
        begin = [t[0] for t in interv]
        evs[event] = begin

    num_samples = [len(evs[ev]) for ev in evs.keys()]
    file_data = {}
    for neuron_str in neurons:
        thisneur = []
        for stim, ts_list in evs.items():
            this_stim = []
            for ts in ts_list:
                neurodata = np.array(neurons[neuron_str])
                spks = np.where((neurodata >= ts) & (neurodata <= ts + t))[0]
                adjusted_spks = neurons[neuron_str][spks] - ts
                this_stim.append(adjusted_spks)
            thisneur.extend(this_stim)
        file_data[neuron_str] = thisneur

    return file_data, num_samples


if "__main__" == __name__:
    tvals = [0.5, 1, 1.5, 2]
    files = {t: {} for t in tvals}

    for t in tvals:
        for file in Path.home().glob("data/rs/*.nex"):
            try:
                data, nsam = get_data(file, t)
                files[t][file.stem] = [data, nsam]
            except Exception as e:
                print(f"Error with {file} at t={t}\n"
                      f"{e}")
                continue

    # %% COMPUTE PAIRWISE DISTANCES
    dists = {t: {} for t in tvals}
    errors = {t: {} for t in tvals}

    for t in tvals:
        for filename, session_data in files[t].items():
            all_neuron_cspks = session_data[0]
            neuron_distances = {}
            for neuron, spikes in all_neuron_cspks.items():
                try:
                    neuron_distances[neuron] = ms.spkd(spikes, qvals)
                except Exception as e:
                    errors[t][filename][neuron] = e
            dists[t][filename] = (neuron_distances, session_data[1])
            print(f"Finished {filename} for t={t}")
    savedir = Path.home() / "data/res"
    save(savedir / "rs_distances.pkl", dists)
