import numpy as np

from src.analysis.model_comparison import (
    layout_similarity, prediction_metrics, select_stable_layout,
)


def gaussian(center, shape=(15, 15)):
    x, y = np.indices(shape)
    return np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / 4).ravel()


def test_prediction_metrics_align_filter_and_use_prediction_count():
    actual = np.array([99, 99, 1.0, 2.0, 4.0])
    predicted = np.array([1.0, 2.0, 3.0])
    metrics = prediction_metrics(actual, predicted, n_parameters=2)
    assert metrics.n_observations == 3
    assert metrics.mse == 1 / 3
    assert np.isfinite(metrics.aic) and np.isfinite(metrics.bic)


def test_layout_similarity_is_order_invariant_and_stability_selects_consensus():
    first = np.stack([gaussian((4, 4)), gaussian((10, 10))])
    reordered = first[::-1].copy()
    outlier = np.stack([gaussian((4, 10)), gaussian((10, 4))])
    assert layout_similarity(first, reordered, (15, 15)) > 0.99
    index, scores, eligible = select_stable_layout(
        [first, reordered, outlier], (15, 15)
    )
    assert index in {0, 1}
    assert set(eligible) == {0, 1, 2}
    assert scores[index] > scores[2]
