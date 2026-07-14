"""Metrics and repeated-run stability analyses recovered from legacy notebooks."""

from dataclasses import dataclass

import numpy as np
import shapely.geometry as geometry
from scipy.optimize import linear_sum_assignment

from .fitellipse import fitellipse


@dataclass(frozen=True)
class PredictionMetrics:
    correlation: float
    mse: float
    aic: float
    bic: float
    n_observations: int
    n_parameters: int


def prediction_metrics(actual, predicted, n_parameters, filter_length=None):
    """Calculate held-out correlation, MSE, AIC, and BIC.

    AIC/BIC use the Gaussian residual likelihood up to an additive constant:
    ``n * log(MSE) + penalty``. This preserves the notebook convention while
    correctly using the number of predicted time bins rather than trial count.
    """
    actual, predicted = np.asarray(actual), np.asarray(predicted)
    if filter_length is None:
        filter_length = len(actual) - len(predicted)
    actual = actual[filter_length:]
    if actual.shape != predicted.shape:
        raise ValueError("aligned actual and predicted responses must have equal shape")
    if not len(predicted):
        raise ValueError("predicted response cannot be empty")
    residual = predicted - actual
    mse = float(np.mean(residual ** 2))
    n = len(predicted)
    if mse <= 0:
        aic = bic = float("-inf")
    else:
        aic = n * np.log(mse) + 2 * n_parameters
        bic = n * np.log(mse) + np.log(n) * n_parameters
    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    return PredictionMetrics(correlation, mse, float(aic), float(bic), n, n_parameters)


def _polygons(layout, crop_shape, sigma):
    layout = np.asarray(layout)
    return [
        geometry.Polygon(fitellipse(row.reshape(crop_shape), sigma=sigma).T)
        for row in layout
    ]


def layout_similarity(first, second, crop_shape, sigma=1.5):
    """Mean optimally matched ellipse Jaccard similarity between two layouts."""
    if len(first) != len(second):
        raise ValueError("layouts must contain the same number of subunits")
    if len(first) == 0:
        return float("nan")
    first_polygons, second_polygons = (
        _polygons(first, crop_shape, sigma), _polygons(second, crop_shape, sigma)
    )
    scores = np.zeros((len(first), len(second)))
    for i, left in enumerate(first_polygons):
        for j, right in enumerate(second_polygons):
            union = left.union(right).area
            scores[i, j] = left.intersection(right).area / union if union else 0
    rows, columns = linear_sum_assignment(scores, maximize=True)
    return float(scores[rows, columns].mean())


def select_stable_layout(layouts, crop_shape, sigma=1.5):
    """Select the most representative layout among runs with the modal size.

    This formalizes the notebook's repeated-training analysis: retain layouts
    with the modal number of subunits, score all pairs by optimal ellipse
    matching, and select the run with the greatest mean similarity.
    """
    if not layouts:
        raise ValueError("at least one layout is required")
    counts = np.asarray([len(layout) for layout in layouts])
    values, frequencies = np.unique(counts, return_counts=True)
    # Preserve the manuscript rule: if multiple counts are equally frequent,
    # prefer the larger number of subunits.
    modal_count = int(values[frequencies == frequencies.max()].max())
    indices = np.flatnonzero(counts == modal_count)
    if len(indices) == 1:
        return int(indices[0]), np.array([1.0]), indices
    similarities = np.eye(len(indices))
    for i in range(len(indices)):
        for j in range(i + 1, len(indices)):
            value = layout_similarity(
                layouts[indices[i]], layouts[indices[j]], crop_shape, sigma
            )
            similarities[i, j] = similarities[j, i] = value
    scores = (similarities.sum(axis=1) - 1) / (len(indices) - 1)
    return int(indices[np.argmax(scores)]), scores, indices
