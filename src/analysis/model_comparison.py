"""Frozen-noise prediction metrics and repeated-run layout stability."""

from dataclasses import dataclass

import numpy as np
import shapely.geometry as geometry
from scipy.optimize import linear_sum_assignment

from .fitellipse import fitellipse


@dataclass(frozen=True)
class PredictionMetrics:
    """How well a model's predicted firing rate matches the measured one.

    Attributes:
        correlation: Pearson correlation of predicted and measured rates.
        mse: Mean squared error (Hz²).
        n_observations: Number of time bins compared.
        normalized_correlation: ``correlation`` divided by the noise ceiling
            (see :func:`correlation_ceiling`); NaN if no ceiling was given.
    """

    correlation: float
    mse: float
    n_observations: int
    normalized_correlation: float


@dataclass(frozen=True)
class NoiseCeiling:
    """The best correlation any model could reach with a noisy measured rate.

    Attributes:
        reliability: Spearman-Brown corrected split-half correlation of the
            trial-averaged response (the fraction of its variance that is
            repeatable signal rather than noise).
        max_correlation: ``sqrt(reliability)``, the expected correlation of
            the true underlying rate with the measured average.
        n_frames: Number of time bins used.
    """

    reliability: float
    max_correlation: float
    n_frames: int


def prediction_metrics(actual, predicted, filter_length=None, ceiling=None):
    """Held-out correlation and mean squared error of a predicted firing rate.

    These, not AIC/BIC, are the comparison between the LN and subunit models:
    both are scored on frozen-noise frames they were not fitted to, and the
    two models' parameter counts are not comparable (the subunit model's
    spatial filters come from a network fitted to other data).

    Args:
        actual: Measured rate for every frame.
        predicted: Prediction for frames ``filter_length`` onwards.
        filter_length: Frames dropped from ``actual`` to align it; defaults to
            the length difference.
        ceiling: Optional noise ceiling (``NoiseCeiling.max_correlation``) for
            ``normalized_correlation``.
    """
    actual, predicted = np.asarray(actual, dtype=float), np.asarray(predicted, dtype=float)
    if filter_length is None:
        filter_length = len(actual) - len(predicted)
    actual = actual[filter_length:]
    if actual.shape != predicted.shape:
        raise ValueError("aligned actual and predicted responses must have equal shape")
    if not len(predicted):
        raise ValueError("predicted response cannot be empty")
    correlation = float(np.corrcoef(actual, predicted)[0, 1])
    normalized = correlation / ceiling if ceiling else float("nan")
    return PredictionMetrics(
        correlation=correlation,
        mse=float(np.mean((predicted - actual) ** 2)),
        n_observations=len(predicted),
        normalized_correlation=float(normalized),
    )


def correlation_ceiling(repeats, filter_length=0, n_splits=200, seed=0):
    """Noise ceiling for correlation with the trial-averaged frozen-noise response.

    The repeats are split at random into two halves ``n_splits`` times; the
    mean correlation between the halves' averages, corrected to the full
    number of repeats with the Spearman-Brown formula ``2r / (1 + r)``, is
    the reliability of the average. Its square root is the correlation a
    perfect model of the underlying rate would be expected to reach.

    Args:
        repeats: Spike counts, frames × repeats (one cell's ``spk2``).
        filter_length: Leading frames to drop, matching the predictions.
        n_splits: Random half splits averaged.
        seed: Seed for the splits.

    Raises:
        ValueError: With fewer than 2 repeats.
    """
    repeats = np.asarray(repeats, dtype=float)[filter_length:]
    n_repeats = repeats.shape[1]
    if n_repeats < 2:
        raise ValueError("a split-half noise ceiling needs at least 2 repeats")
    rng = np.random.default_rng(seed)
    correlations = []
    for _ in range(n_splits):
        order = rng.permutation(n_repeats)
        half = n_repeats // 2
        first, second = repeats[:, order[:half]].mean(axis=1), repeats[:, order[half:]].mean(axis=1)
        correlations.append(np.corrcoef(first, second)[0, 1])
    r = float(np.nanmean(correlations))
    reliability = max(0.0, 2 * r / (1 + r)) if r > -1 else 0.0
    return NoiseCeiling(reliability, float(np.sqrt(reliability)), len(repeats))


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

    Keeps the layouts with the most common number of subunits (the larger
    count on a tie), scores every pair by optimally matched ellipse Jaccard
    similarity (:func:`layout_similarity`), and returns the run with the
    greatest mean similarity to the others. The Colab notebook used the same
    idea but resolved ties toward the smaller count (``scipy.stats.mode``)
    and matched subunits greedily, so it can pick a different run.

    Returns:
        (index of the selected layout, mean similarity of each eligible
        layout, indices of the eligible layouts).
    """
    if not layouts:
        raise ValueError("at least one layout is required")
    counts = np.asarray([len(layout) for layout in layouts])
    values, frequencies = np.unique(counts, return_counts=True)
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
