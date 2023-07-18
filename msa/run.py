from pw_matrix import get_anear
from pw_info import get_info
from pw_distances import get_data
from pathlib import Path
import metricspace as ms
from data_utils.data_io import load, save
import numpy as np
print(Path.home())
qvals = np.concatenate(([0], 2 ** np.arange(-4, 9.5, 0.5)))

if "__main__" == __name__:
    # data = load("/home/flynn/data/res/distances.pkl")
    home = Path.home()

    data = load(home / "Dropbox" / "Lab" / "SF" / "data" / "save.pkl")
    tvals = [0.5, 1, 1.5, 2]
    anear = {t: {filename: {} for filename, _ in data[t].items()} for t in tvals}
    errors = []

    for t in tvals:
        for filename, filedata in data[t].items():
            neuron_dists = filedata[0]
            nsam = filedata[1]
            for neuron, dists in neuron_dists.items():
                # try:
                anear_mat = get_anear(dists, nsam)
                # except Exception as e:
                #     errors.append((t, filename, neuron, e))
                #     print(f"Error with {filename} {neuron} at t={t}")
                #     continue
                anear[t][filename][neuron] = anear_mat

    x = 3
