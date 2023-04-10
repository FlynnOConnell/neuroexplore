from __future__ import annotations

import logging
import numpy as np
import matplotlib
from typing import Iterable, Optional, Any, Tuple
from matplotlib import rcParams, lines

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(name)s - %(message)s")


def get_axis_labels(ax, data):
    ax.set_xlabel(data.columns[0], weight="bold")
    ax.set_ylabel(data.columns[1], weight="bold")
    if data.shape[1] > 2:
        ax.set_zlabel(data.columns[1], weight="bold")
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
    proxy, label = get_handles_from_dict(
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


def get_handles_from_dict(
        color_dict: dict,
        markersize: Optional[int] = 5,
        marker: Optional[str] = 'o',
        **kwargs
) -> Tuple[list, list]:
    """
    Get matplotlib handles for input dictionary.
    Args:
        markersize ():
        color_dict (dict): Dictionary of event:color k/v pairs.
        marker (str): Shape of scatter point, default is circle.
    Returns:
        proxy (list): matplotlib.lines.line2D appended list.
        label (list): legend labels for each proxy.
    """
    proxy, label = [], []
    for t, c in color_dict.items():
        proxy.append(
            lines.Line2D(
                [0],
                [0],
                marker=marker,
                markersize=markersize,
                markerfacecolor=c,
                markeredgecolor="None",
                linestyle="None",
                **kwargs
            ))
        label.append(t)
    return proxy, label


def get_legend(ax, color_dict, colors, markersize):
    mydict = {k: v for k, v in color_dict.items() if color_dict[k] in np.unique(colors)}
    proxy, label = get_handles_from_dict(mydict, markersize)
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
