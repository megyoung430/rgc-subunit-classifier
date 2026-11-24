import numpy as np
import pytest

from src.prediction.sta_model_predictions import separable_ln_filter


def test_matches_the_cropped_sta_layout_for_a_separable_sta():
    rng = np.random.default_rng(0)
    spatial, temporal = rng.normal(size=(6, 5)), rng.normal(size=4)
    sta = spatial[:, :, None] * temporal
    crop = (slice(1, 4), slice(2, 5))
    mask = np.ones((3, 3), dtype=bool)
    expected = sta[crop].reshape(9, 4)
    np.testing.assert_allclose(separable_ln_filter(spatial, temporal, crop, mask), expected)


def test_pixels_outside_the_mask_have_zero_weight():
    spatial, temporal = np.ones((4, 4)), np.array([1.0, -1.0])
    mask = np.eye(2, dtype=bool)
    weights = separable_ln_filter(spatial, temporal, (slice(0, 2), slice(0, 2)), mask)
    np.testing.assert_array_equal(weights[~mask.ravel()], 0)


def test_mask_must_match_the_crop():
    with pytest.raises(ValueError, match="mask has shape"):
        separable_ln_filter(np.ones((4, 4)), np.ones(2), (slice(0, 2), slice(0, 2)),
                            np.ones((3, 3), dtype=bool))
