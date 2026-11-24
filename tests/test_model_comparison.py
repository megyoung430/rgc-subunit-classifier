import numpy as np
import pytest

from src.analysis.model_comparison import (
    correlation_ceiling,
    layout_similarity,
    prediction_metrics,
    select_stable_layout,
)


def gaussian(center, shape=(15, 15)):
    x, y = np.indices(shape)
    return np.exp(-((x - center[0]) ** 2 + (y - center[1]) ** 2) / 4).ravel()


def test_prediction_metrics_align_the_filter_and_report_correlation_and_mse():
    actual = np.array([99, 99, 1.0, 2.0, 4.0])
    predicted = np.array([1.0, 2.0, 3.0])
    metrics = prediction_metrics(actual, predicted)
    assert metrics.n_observations == 3
    assert metrics.mse == 1 / 3
    assert metrics.correlation == pytest.approx(np.corrcoef([1, 2, 4], [1, 2, 3])[0, 1])
    assert np.isnan(metrics.normalized_correlation)
    assert not hasattr(metrics, "aic")
    scaled = prediction_metrics(actual, predicted, ceiling=0.5)
    assert scaled.normalized_correlation == pytest.approx(2 * metrics.correlation)


def poisson_repeats(n_repeats, seed=0, frames=3000):
    rng = np.random.default_rng(seed)
    rate = np.exp(np.convolve(rng.normal(size=frames), np.ones(5) / 5, "same"))
    return rate, rng.poisson(rate[:, None], size=(frames, n_repeats))


def test_correlation_ceiling_matches_the_true_rates_achievable_correlation():
    rate, repeats = poisson_repeats(20)
    ceiling = correlation_ceiling(repeats, seed=0)
    achievable = np.corrcoef(rate, repeats.mean(axis=1))[0, 1]
    assert ceiling.max_correlation == pytest.approx(achievable, abs=0.03)
    assert 0 < ceiling.reliability < 1
    perfect = prediction_metrics(repeats.mean(axis=1), rate, filter_length=0,
                                 ceiling=ceiling.max_correlation)
    assert perfect.normalized_correlation == pytest.approx(1.0, abs=0.04)


def test_correlation_ceiling_drops_the_filter_frames_and_needs_two_repeats():
    _, repeats = poisson_repeats(6, frames=200)
    assert correlation_ceiling(repeats, filter_length=20).n_frames == 180
    with pytest.raises(ValueError, match="at least 2 repeats"):
        correlation_ceiling(repeats[:, :1])


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


def test_ceiling_without_any_defined_split_is_an_error():
    with pytest.raises(ValueError, match="noise ceiling"):
        correlation_ceiling(np.zeros((50, 4)))
