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
    anear = {}
    data = load("/home/flynn/repos/neuroexplore/msa/data/res/repstim_distances_0707.pkl")
    for filename, filedata in data.items():
        neuron_dists = filedata[0]
        nsam = filedata[1]
        for neuron, dists in neuron.items():
            anear_mat = get_anear(dists, nsam)
            anear[filename][neuron] = anear_mat