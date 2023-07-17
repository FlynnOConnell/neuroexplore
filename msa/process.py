from __future__ import annotations
import numpy as np
import pandas as pd
from data_utils.data_io import load, save
from analysis.output import Output

qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))
data = pd.DataFrame(columns=["filename", "neuron", "q", "info", "ja"])


def process(data_info, t):
    list_df_each_neuron = []
    final_df = pd.DataFrame(columns=["filename", "neuron", "qmax", "h", "h_corrected"])
    for filename, neurondict in data_info.items():
        for neuron, info_ja in neurondict.items():
            qmaxidx = np.argmax(info_ja[0])
            qmax_val = qvals[qmaxidx]
            hmax_val = info_ja[0][qmaxidx]
            bias_val = info_ja[1][qmaxidx]
            bias = hmax_val + bias_val
            neuron_frame = pd.DataFrame(
                [{"filename": filename, "neuron": neuron, "qmax": qmax_val, "h": hmax_val, "h_corrected": bias}],
            )
            list_df_each_neuron.append(neuron_frame)
        final_df = pd.concat(list_df_each_neuron, ignore_index=True)
    return final_df.sort_values(by=["filename"])


if __name__ == "__main__":
    info = load("/home/flynn/data/res/rs_info.pkl")
    max_only = process(info, 2.0)
    output = Output(max_only, "/home/flynn/data/res/output")
    output.save_file("temporal_coding", "rs_max")

    x = 5
