from __future__ import annotations
import numpy as np
from pathlib import Path
import metricspace as ms
from data_utils.io import load, save


def get_anear(input_dists, numsamples):
    nq = 28
    q_anear = np.zeros((len(numsamples), len(numsamples), nq))
    for q in range(nq):
        q_anear[:, :, q] = ms.distclust(input_dists[:, :, q], numsamples)
    return q_anear


if __name__ == "__main__":
    home = Path.home()
    anear = {}
    data = load(home / "data/rs/respstim_dists_0705.pkl")
    for filename, filedata in data.items():
        for neuron in filedata.keys():
            pw_dists = filedata[neuron]
            n_samples = filedata['n_samples']
            anear_mat = get_anear(pw_dists, n_samples)
            anear[filename] = anear_mat

