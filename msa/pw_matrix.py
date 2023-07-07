
from __future__ import annotations
import numpy as np
from pathlib import Path
import metricspace as ms
from data_utils.io import load, save


def get_anear(pw_dists, n_samples):
    nq = 28
    anear_mat = np.zeros((len(n_samples), len(n_samples), nq))
    for q in range(nq):
        anear_mat[:, :, q] = ms.distclust(pw_dists[:, :, q], n_samples)
    return anear_mat


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

