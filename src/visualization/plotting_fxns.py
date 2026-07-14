"""Small, side-effect-free plotting helpers distilled from plotting notebooks."""

import math

import numpy as np
from matplotlib import pyplot as plt


def plot_subunits(subunits, crop_shape, *, columns=3, cmap="bwr"):
    """Plot spatial weights with symmetric per-subunit color limits."""
    subunits = np.asarray(subunits)
    rows = max(1, math.ceil(len(subunits) / columns))
    figure, axes = plt.subplots(rows, columns, squeeze=False)
    for index, axis in enumerate(axes.flat):
        if index >= len(subunits):
            axis.set_visible(False)
            continue
        image = subunits[index].reshape(crop_shape)
        limit = np.abs(image).max() or 1
        axis.imshow(image.T, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
        axis.set_title(f"Subunit {index}")
        axis.set_axis_off()
    figure.tight_layout()
    return figure, axes


def plot_predictions(predicted, actual, time, filter_length, *, label="Model", ax=None):
    """Overlay observed and predicted frozen-stimulus firing rates."""
    if ax is None:
        _, ax = plt.subplots()
    ax.plot(time[filter_length:], np.asarray(actual)[filter_length:], label="Actual")
    ax.plot(time[filter_length:], predicted, label=label)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing rate (Hz)")
    ax.legend()
    return ax
