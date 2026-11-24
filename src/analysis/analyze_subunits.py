"""Select subunits from network weights by spatial smoothness, and measure their size."""

import math

import numpy as np

from .fitellipse import fitellipse

# Stimulus pixel size in micrometres on the retina.
PIXEL_SIZE_UM = 30


def morans_i(image, mask=None):
    """Moran's I (4-neighbour spatial autocorrelation) of a 2-D weight map.

    Near 0 for spatially white noise and close to 1 for a smooth blob. With
    ``mask``, only pixels inside the mask count: the mean is taken over them
    and only neighbour pairs with both pixels inside contribute. Without it,
    every pixel of ``image`` counts. Use the ellipse mask for network weights,
    which are zero outside it by construction: counting those zeros would
    make any weight map with a consistent sign look smooth.

    Returns:
        Moran's I, or NaN if the weights inside the mask are all equal (for
        example, a node whose weights were all driven to zero).
    """
    image = np.asarray(image, dtype=float)
    mask = np.ones(image.shape, dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    if mask.shape != image.shape:
        raise ValueError(f"mask has shape {mask.shape}; image has shape {image.shape}")
    if not mask.any():
        raise ValueError("mask selects no pixels")
    deviations = np.where(mask, image - image[mask].mean(), 0.0)
    products = pairs = 0.0
    for a, b, both in (
        (deviations[:-1, :], deviations[1:, :], mask[:-1, :] & mask[1:, :]),
        (deviations[:, :-1], deviations[:, 1:], mask[:, :-1] & mask[:, 1:]),
    ):
        products += 2 * (a * b)[both].sum()  # each neighbour pair counted both ways
        pairs += 2 * both.sum()
    squares = (deviations[mask] ** 2).sum()
    if squares == 0 or pairs == 0:
        return float("nan")
    return float(mask.sum() * products / pairs / squares)


def morans_i_values(weights, crop_shape, mask=None):
    """Moran's I of each row of ``weights`` (nodes × crop pixels) on the crop grid."""
    return np.array([morans_i(row.reshape(crop_shape), mask) for row in np.asarray(weights)])


def get_morans_i(subunit, crop_x, crop_y, mask=None):
    """Moran's I of one subunit, given as a ``crop_x × crop_y`` (or flat) array."""
    return morans_i(np.asarray(subunit).reshape(crop_x, crop_y), mask)


def select_by_morans_i(weights, values, threshold_morans_i):
    """Rows of ``weights`` whose Moran's I in ``values`` exceeds the threshold.

    NaN values (no weight variation) are never selected.
    """
    weights, values = np.asarray(weights), np.asarray(values)
    if len(values) != len(weights):
        raise ValueError(f"{len(values)} Moran's I values for {len(weights)} weight rows")
    return weights[np.nan_to_num(values, nan=-np.inf) > threshold_morans_i]


def get_subunits(weights, crop_x, crop_y, threshold_morans_i=0.25, mask=None):
    """Rows of ``weights`` whose Moran's I exceeds the threshold.

    Args:
        weights: Hidden-node (or encoder) weights, nodes × ``crop_x * crop_y``.
        crop_x, crop_y: Size of the cropped window.
        threshold_morans_i: Nodes with Moran's I above this are subunits.
        mask: Optional boolean ``crop_x × crop_y`` ellipse mask; Moran's I is
            then computed on the pixels inside it only.

    Returns:
        The selected rows, subunits × pixels. Nodes whose Moran's I is NaN
        (no weight variation inside the mask) are never selected.
    """
    values = morans_i_values(weights, (crop_x, crop_y), mask)
    return select_by_morans_i(weights, values, threshold_morans_i)


def _effective_diameter(image, pixel_size_um):
    """Diameter (µm) of the circle with the area of the 1.5σ Gaussian ellipse."""
    _, _, d_major, d_minor, _ = fitellipse(image, sigma=1.5, raw=True)
    area = math.pi * (d_major / 2) * (d_minor / 2)
    return pixel_size_um * 2 * math.sqrt(area / math.pi)


def get_subunit_sizes(subunits, crop_x, crop_y, ignore_negative=True,
                      pixel_size_um=PIXEL_SIZE_UM):
    """Effective diameter of each subunit in micrometres.

    Args:
        subunits: Subunits × ``crop_x * crop_y`` weights.
        crop_x, crop_y: Size of the cropped window.
        ignore_negative: If true, subunits with negative mean weight get NaN.
        pixel_size_um: Stimulus pixel size on the retina.

    Returns:
        One diameter per subunit, in input order (NaN where ignored).
    """
    subunits = np.asarray(subunits)
    sizes = np.full(len(subunits), np.nan)
    for index, row in enumerate(subunits):
        image = row.reshape(crop_x, crop_y)
        if ignore_negative and image.mean() < 0:
            continue
        sizes[index] = _effective_diameter(image, pixel_size_um)
    return sizes


def get_rf_size(rf, pixel_size_um=PIXEL_SIZE_UM):
    """Effective diameter (µm) of a receptive field from its 1.5σ ellipse."""
    return _effective_diameter(np.asarray(rf), pixel_size_um)
