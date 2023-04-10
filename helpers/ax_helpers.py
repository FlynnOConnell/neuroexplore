from __future__ import annotations

import logging
from typing import Optional, Any
import numpy as np
from helpers import funcs
import matplotlib

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


def get_axis_labels(ax, data):
    ax.set_xlabel(data.columns[0], weight="bold")
    ax.set_ylabel(data.columns[1], weight="bold")
    if data.shape[1] > 2:
        ax.set_zlabel(data.columns[1], weight="bold")
    return ax


def get_legend(ax, color_dict, colors, markersize):
    mydict = {k: v for k, v in color_dict.items() if color_dict[k] in np.unique(colors)}
    proxy, label = funcs.get_handles_from_dict(mydict, markersize)
    ax.legend(
        handles=proxy,
        labels=label,
        prop={"size": 20},
        bbox_to_anchor=(1.05, 1),
        ncol=2,
        numpoints=1,
        facecolor=ax.get_facecolor(),
        edgecolor=None,
        fancybox=True,
        markerscale=True)
    return ax


def make_legend(
        mydict: dict,
        marker: Optional[str] = 'o',
        markeralpha=None,
) -> (matplotlib.pyplot.figure, matplotlib.axes.Axes):
    import pylab
    fig = pylab.figure()
    figlegend = pylab.figure(figsize=(3, 2))
    ax = fig.add_subplot(111)
    proxy, label = funcs.get_handles_from_dict(
        mydict,
        markersize=5,
        marker=marker
    )
    leg = figlegend.legend(
        handles=proxy,
        labels=label,
        fancybox=False,
    )
    if markeralpha is not None:
        for lh in leg.legendHandles:
            lh.set_alpha(markeralpha)

    return fig, ax