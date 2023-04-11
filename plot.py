import numpy as np
import pandas as pd
import seaborn as sb
import matplotlib.pyplot as plt
from matplotlib import rcParams, lines
from helpers import funcs, ax_helpers


rcParams.update(
    {
        "font.weight": "bold",
        "font.family": "Arial",
        'font.sans-serif': "Arial",
        "axes.labelweight": "bold",
        "xtick.major.width": "1.3",
        "axes.facecolor": "w",
        "axes.labelsize": 10,
        "lines.linewidth": 1,
    }
)

class Plot:
    """
    Parameters:
    ___________
    """

    def __init__(
            self,
            data,
    ) -> None:

        # TODO: glob for .csv and .xlxs
        self.data = data


    def histogram(self, binsize) -> None:

        a_trial = self.data['e_BR'][4]
        bins = np.arange(a_trial.Neuron.iloc[0], a_trial.Neuron.iloc[-1], binsize)
        y_height, x_bins = np.histogram(a_trial.Neuron, bins.reshape(-1))
        barcol = [funcs.get_matched_time(np.asarray(a_trial.Neuron), x)[0] for x in x_bins]
        a_ev = a_trial.Color.iloc[[np.where(a_trial.Neuron == x)[0][0] for x in barcol]]

        thisdict = {'Peanuts': 'orange',
                    'Well': 'blue',
                    'Movement': 'black'
                    }
        fig, ax = plt.subplots()
        ax.bar(x_bins[:-1], y_height, width=binsize, color=a_ev)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1)
        proxy, label = ax_helpers.get_handles_from_dict(thisdict, 3, marker="s")
        ax.legend(
            handles=proxy,
            labels=label,
            facecolor=ax.get_facecolor(),
            edgecolor=None,
            fancybox=False,
            markerscale=3,
            shadow=True)

        ax.set_title("Frequency Histogram", fontweight='bold')
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Spikes / second", fontweight='bold')

        plt.tight_layout()
        plt.show()