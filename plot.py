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
            spikes,
            events,
            colors
    ) -> None:

        # TODO: glob for .csv and .xlxs
        self.spikes = spikes
        self.events = events
        self.colors = colors
        self._validate()

    def _validate(self):
        if not len(self.spikes) == len(self.events) == len(self.colors):
            raise Exception(f'Input arrays must be equal length. \n '
                            f'------------------------------------------ \n'
                            f'Spikes: {len(self.spikes)}, '
                            f'Events: {len(self.events)}, '
                            f'Colors: {len(self.colors)}')

    def histogram(self, binsize: int | float = 0.1, legend_dict: dict = None) -> None:
        bins = np.arange(self.spikes.iloc[0], self.spikes.iloc[-1], binsize)
        y_height, x_bins = np.histogram(self.spikes, bins.reshape(-1))
        barcol = [funcs.get_matched_time(np.asarray(self.spikes), x)[0] for x in x_bins]
        self.colors = self.colors.reset_index(drop=True)
        a_ev = self.colors[[np.where(self.spikes == x)[0][0] for x in barcol]]

        fig, ax = plt.subplots()
        ax.bar(x_bins[:-1], y_height, width=.1, color=a_ev)
        ax.set_ylim(ax.get_ylim()[0], ax.get_ylim()[1] + 1)
        if legend_dict:
            proxy, label = ax_helpers.get_handles_from_dict(legend_dict, 3, marker="s")
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

    def isi_histogram(self, time, binsize: int | float = 0.1, legend_dict: dict = None) -> None:

        bins = np.arange(0, .5, 1e-3)  # Define the bins for the histogram
        plt.hist(self.spikes, bins)  # Plot the histogram of the ISI data
        plt.xlim([0, 0.15])  # ... focus on ISIs from 0 to 150 ms
        plt.xlabel('ISI [s]')  # ... label the x-axis
        plt.ylabel('Counts')  # ... and the y-axis
        plt.title('Low-light')  # ... give the plot a title
        plt.show()
