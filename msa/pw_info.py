from __future__ import annotations
import numpy as np
import metricspace as ms
from data_utils.data_io import load, save


def get_info(conf_matrix):
    nq = 28
    h_info = np.zeros(nq)
    mat_ja = np.zeros(nq)
    for q in range(nq):
        total = np.sum(conf_matrix[:, :, q])
        mat_prob = conf_matrix[:, :, q] / total
        h_info[q] = ms.tblxinfo(mat_prob)
        mat_ja[q] = ms.tblxbi(conf_matrix[:, :, q], 'ja', None)
    return h_info, mat_ja


if __name__ == "__main__":
    anear = load("/home/flynn/data/res/anear.pkl")
    all_info = {filename: {} for filename, _ in anear.items()}
    for filename, neurondict in anear.items():
        for neuron, conf_mat in neurondict.items():
            info, ja = get_info(conf_mat)
            all_info[filename][neuron] = [info, ja]

    save("/home/flynn/data/res/info.pkl", all_info)
