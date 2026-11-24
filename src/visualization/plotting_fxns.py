"""Plotting helpers: each draws into a given axis (or makes one) and returns it.

Figures follow one style (:func:`use_style`): Arial, Okabe–Ito colours,
frameless legends placed outside the data, and title-case labels.
:func:`save_figure` writes every figure as pdf, png and svg with an optional
``<name>.caption.txt`` beside it for the longer explanation.
"""

import math
from pathlib import Path

import numpy as np
from matplotlib import pyplot as plt

from ..outputs import check_output_dir

# Okabe & Ito (2008) colour-blind-safe palette.
OKABE_ITO = {
    "black": "#000000", "orange": "#E69F00", "sky blue": "#56B4E9",
    "green": "#009E73", "yellow": "#F0E442", "blue": "#0072B2",
    "vermillion": "#D55E00", "purple": "#CC79A7",
}
MODEL_COLORS = {
    "Measured": OKABE_ITO["black"],
    "LN": OKABE_ITO["orange"],
    "Classifier Subunit": OKABE_ITO["blue"],
    "Autoencoder Subunit": OKABE_ITO["vermillion"],
}


def use_style():
    """Set the shared figure style for the rest of the session."""
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "axes.spines.top": False,
        "axes.spines.right": False,
        "legend.frameon": False,
        "pdf.fonttype": 42,  # keep text editable in Illustrator
        "svg.fonttype": "none",
    })


def outside_legend(ax, **kwargs):
    """Frameless legend to the right of the data."""
    return ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1), **kwargs)


def save_figure(fig, path_stem, caption=None, dpi=200):
    """Save ``fig`` as ``<path_stem>.pdf``, ``.png`` and ``.svg``.

    ``caption`` (optional) is written to ``<path_stem>.caption.txt``. Files are
    overwritten on re-runs, but never inside the Colab archive.

    Returns:
        The list of files written.
    """
    stem = Path(path_stem)
    folder = check_output_dir(stem.parent)
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for suffix in (".pdf", ".png", ".svg"):
        path = stem.with_name(stem.name + suffix)
        fig.savefig(path, dpi=dpi, bbox_inches="tight")
        written.append(path)
    if caption is not None:
        path = stem.with_name(stem.name + ".caption.txt")
        path.write_text(caption.rstrip("\n") + "\n")
        written.append(path)
    return written


def plot_subunits(subunits, crop_shape, *, axes=None, columns=3, cmap="bwr",
                  titles=None, mask=None):
    """Spatial weight maps, one per row of ``subunits``, with symmetric colour limits.

    Args:
        subunits: Subunits (or nodes) × pixels.
        crop_shape: Shape of the cropped window.
        axes: Optional array of axes to draw into; spare axes are hidden.
        columns: Columns of the grid made when ``axes`` is not given.
        cmap: Diverging colormap; white is zero weight.
        titles: Optional title per map (default "Subunit <i>").
        mask: Optional ellipse mask, outlined on each map.

    Returns:
        (figure, axes).
    """
    subunits = np.asarray(subunits)
    if axes is None:
        rows = max(1, math.ceil(len(subunits) / columns))
        figure, axes = plt.subplots(rows, columns, squeeze=False,
                                    figsize=(2.2 * columns, 2.2 * rows))
    axes = np.atleast_1d(axes)
    figure = axes.flat[0].figure
    for index, axis in enumerate(axes.flat):
        if index >= len(subunits):
            axis.set_visible(False)
            continue
        image = subunits[index].reshape(crop_shape)
        limit = np.abs(image).max() or 1
        axis.imshow(image.T, origin="lower", cmap=cmap, vmin=-limit, vmax=limit)
        if mask is not None:
            axis.contour(np.asarray(mask, dtype=float).T, levels=[0.5], colors="0.5",
                         linewidths=0.6)
        axis.set_title(titles[index] if titles is not None else f"Subunit {index}", fontsize=9)
        axis.set_axis_off()
    return figure, axes


def plot_predictions(predicted, actual, time, filter_length, *, label="Model",
                     ax=None, color=None):
    """Measured and predicted frozen-noise firing rates over time."""
    if ax is None:
        _, ax = plt.subplots(figsize=(9, 3))
    ax.plot(time[filter_length:], np.asarray(actual)[filter_length:],
            color=MODEL_COLORS["Measured"], label="Measured", linewidth=1)
    ax.plot(time[filter_length:], predicted, label=label, linewidth=1.2,
            color=color or MODEL_COLORS.get(label))
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Firing Rate (Hz)")
    outside_legend(ax)
    return ax


def plot_crop(spatial_sta, crop, mask, ax=None):
    """Spatial STA with the crop box and the ellipse mask outlined."""
    if ax is None:
        _, ax = plt.subplots(figsize=(5, 4))
    spatial_sta = np.asarray(spatial_sta)
    limit = np.abs(spatial_sta).max() or 1
    ax.imshow(spatial_sta.T, origin="lower", cmap="bwr", vmin=-limit, vmax=limit)
    full = np.zeros(spatial_sta.shape)
    full[crop] = mask
    ax.contour(full.T, levels=[0.5], colors=OKABE_ITO["green"], linewidths=1)
    (x0, x1), (y0, y1) = (crop[0].start, crop[0].stop), (crop[1].start, crop[1].stop)
    ax.add_patch(plt.Rectangle((x0 - 0.5, y0 - 0.5), x1 - x0, y1 - y0, fill=False,
                               edgecolor=OKABE_ITO["black"], linewidth=0.8))
    ax.set_xlabel("X (Pixels)")
    ax.set_ylabel("Y (Pixels)")
    return ax


def plot_nonlinearity(centers, rates, curve, ax=None, color=None, label=None):
    """Binned drive vs firing rate (points) and the fitted nonlinearity (line).

    Args:
        centers, rates: Mean drive and firing rate per bin.
        curve: Function mapping drive to rate (the fitted nonlinearity).
    """
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    ax.plot(centers, rates, "o", markersize=3, color=color, alpha=0.8)
    grid = np.linspace(np.min(centers), np.max(centers), 200)
    ax.plot(grid, curve(grid), color=color, label=label)
    ax.set_xlabel("Linear Drive")
    ax.set_ylabel("Firing Rate (Hz)")
    return ax


def plot_training_curves(losses, ax=None, ylabel="Loss"):
    """Training and validation loss per epoch."""
    if ax is None:
        _, ax = plt.subplots(figsize=(4, 3))
    epochs = np.arange(1, len(losses["train"]) + 1)
    ax.plot(epochs, losses["train"], color=OKABE_ITO["sky blue"], label="Train")
    if losses.get("valid"):
        ax.plot(epochs[:len(losses["valid"])], losses["valid"], color=OKABE_ITO["blue"],
                label="Validation")
    ax.set_xlabel("Epoch")
    ax.set_ylabel(ylabel)
    outside_legend(ax)
    return ax
