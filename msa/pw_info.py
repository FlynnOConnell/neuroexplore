from __future__ import annotations
from helpers.funcs import interval
import numpy as np
from pathlib import Path
from nex.nexfile import Reader
import metricspace as ms
from data_utils.io import load, save


def get_info(conf_matrix):
    nq = 28
    h_info = np.zeros(nq)
    mat_ja = np.zeros(nq)
    mat_tp = np.zeros(nq)
    for q in range(nq):
        total = np.sum(conf_matrix[:, :, q])
        mat_prob = conf_matrix[:, :, q] / total
        h_info[q] = ms.tblxinfo(mat_prob)
        mat_ja[q] = ms.tblxbi(conf_matrix[:, :, q], 'ja', None)
        mat_tp[q] = ms.tblxbi(conf_matrix[:, :, q], 'tr', 0)
    return h_info, mat_ja, mat_tp


anear = load("/home/flynn/data/res/anear.pkl")
for filename, neurondict in anear.items():
    for neuron, conf_mat in neurondict.items():
        info, ja, tp = get_info(conf_mat)
        print(f"{filename} {neuron} {info} {ja} {tp}")


x=5