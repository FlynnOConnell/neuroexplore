from __future__ import annotations
import numpy as np
import metricspace as ms
from data_utils.data_io import load, save


def get_anear(input_dists, numsamples):
    nq = 28
    q_anear = np.zeros((len(numsamples), len(numsamples), nq))
    for q in range(nq):
        q_anear[:, :, q] = ms.distclust(input_dists[:, :, q], numsamples)
    return q_anear


if __name__ == "__main__":
    errors = []
    data = load("/home/flynn/data/res/distances.pkl")
    anear = {filename: {} for filename, _ in data.items()}
    for filename, filedata in data.items():
        neuron_dists = filedata[0]
        nsam = filedata[1]
        for neuron, dists in neuron_dists.items():
            try:
                anear_mat = get_anear(dists, nsam)
            except Exception as e:
                errors.append((filename, neuron, e))
                print(f"Error with {filename} {neuron}")
                continue
            anear[filename][neuron] = anear_mat
    save("/home/flynn/data/res/anear.pkl", anear)
    x = 5
