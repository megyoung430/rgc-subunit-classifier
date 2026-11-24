import numpy as np
import pytest

from src.prediction.sta_model_predictions import fit_softplus, nonlin, separable_ln_filter


def sliced_like_gen_sta(sta):
    """Spatial and temporal slices through the STA's largest absolute value, unit norm."""
    x, y, t = np.unravel_index(np.abs(sta).argmax(), sta.shape)
    spatial, temporal = sta[:, :, t].copy(), sta[x, y, :].copy()
    return spatial / np.linalg.norm(spatial), temporal / np.linalg.norm(temporal)


def test_matches_the_cropped_sta_layout_for_a_separable_sta():
    rng = np.random.default_rng(0)
    sta = rng.normal(size=(6, 5))[:, :, None] * rng.normal(size=4)
    spatial, temporal = sliced_like_gen_sta(sta)
    crop = (slice(1, 4), slice(2, 5))
    weights = separable_ln_filter(spatial, temporal, crop, np.ones((3, 3), dtype=bool))
    expected = sta[crop].reshape(9, 4)
    ratio = weights / expected
    assert np.allclose(ratio, ratio.flat[0]) and ratio.flat[0] > 0


def test_pixels_outside_the_mask_have_zero_weight():
    spatial, temporal = np.ones((4, 4)), np.array([1.0, -1.0])
    mask = np.eye(2, dtype=bool)
    weights = separable_ln_filter(spatial, temporal, (slice(0, 2), slice(0, 2)), mask)
    np.testing.assert_array_equal(weights[~mask.ravel()], 0)


def test_mask_must_match_the_crop():
    with pytest.raises(ValueError, match="mask has shape"):
        separable_ln_filter(np.ones((4, 4)), np.ones(2), (slice(0, 2), slice(0, 2)),
                            np.ones((3, 3), dtype=bool))


@pytest.mark.parametrize("polarity", [1, -1])
def test_filter_has_the_sign_of_the_sta_for_on_and_off_cells(polarity):
    rng = np.random.default_rng(2)
    blob = np.exp(-((np.arange(6)[:, None] - 3) ** 2 + (np.arange(5)[None, :] - 2) ** 2) / 2)
    sta = polarity * blob[:, :, None] * np.array([0.1, -0.3, 1.0, 0.4])
    sta += 0.01 * rng.normal(size=sta.shape)
    spatial, temporal = sliced_like_gen_sta(sta)
    crop = (slice(1, 5), slice(0, 5))
    weights = separable_ln_filter(spatial, temporal, crop, np.ones((4, 5), dtype=bool))
    assert np.corrcoef(weights.ravel(), sta[crop].reshape(20, 4).ravel())[0, 1] > 0.9


def test_softplus_fit_recovers_parameters_and_rejects_non_finite_input():
    centers = np.linspace(-2, 2, 30)
    np.testing.assert_allclose(fit_softplus(centers, nonlin(centers, 5.0, 2.0, 0.5)),
                               [5.0, 2.0, 0.5], rtol=1e-4)
    with pytest.raises(ValueError, match="finite"):
        fit_softplus(centers, np.full(30, np.nan))
