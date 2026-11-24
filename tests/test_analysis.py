import numpy as np
import pytest

from src.analysis.analyze_subunits import (
    get_morans_i,
    get_rf_size,
    get_subunit_sizes,
    get_subunits,
    morans_i,
)
from src.analysis.fitellipse import fitellipse


def reference_morans_i(image):
    """The original double loop over every pixel of the crop (4-neighbour)."""
    z = image - image.mean()
    products = weights = 0.0
    nx, ny = image.shape
    for x in range(nx):
        for y in range(ny):
            for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                if 0 <= x + dx < nx and 0 <= y + dy < ny:
                    products += z[x, y] * z[x + dx, y + dy]
                    weights += 1
    return image.size * products / weights / (z**2).sum()


def disc_mask(shape, radius):
    x, y = np.indices(shape)
    cx, cy = (shape[0] - 1) / 2, (shape[1] - 1) / 2
    return (x - cx) ** 2 + (y - cy) ** 2 <= radius**2


def test_fitellipse_recovers_gaussian_center():
    x, y = np.indices((21, 17))
    image = np.exp(-((x - 10) ** 2 / 8 + (y - 7) ** 2 / 4))
    center_x, center_y, major, minor, _ = fitellipse(image, raw=True)
    np.testing.assert_allclose([center_x, center_y], [10, 7], atol=0.1)
    assert major >= minor > 0


def test_without_a_mask_morans_i_matches_the_original_loop():
    rng = np.random.default_rng(0)
    for shape in ((3, 3), (7, 5), (12, 9)):
        image = rng.normal(size=shape)
        assert morans_i(image) == pytest.approx(reference_morans_i(image))


def test_noise_inside_the_mask_is_not_selected_even_with_an_offset():
    rng = np.random.default_rng(1)
    mask = disc_mask((24, 24), 10)
    values = []
    for _ in range(200):
        image = np.zeros(mask.shape)
        image[mask] = rng.normal(1.0, 1.0, mask.sum())  # all-positive offset noise
        values.append(morans_i(image, mask))
    assert abs(np.mean(values)) < 0.05
    assert np.mean(np.array(values) > 0.25) == 0


def test_a_smooth_blob_inside_the_mask_scores_high():
    mask = disc_mask((24, 24), 10)
    x, y = np.indices(mask.shape)
    blob = np.exp(-((x - 9) ** 2 + (y - 13) ** 2) / 12.0)
    assert morans_i(np.where(mask, blob, 0), mask) > 0.7


def test_subunit_selection_uses_the_threshold_and_the_mask():
    smooth = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=float)
    checker = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert get_morans_i(smooth, 3, 3) > get_morans_i(checker, 3, 3)
    weights = np.stack([smooth.ravel(), checker.ravel()])
    np.testing.assert_array_equal(get_subunits(weights, 3, 3, threshold_morans_i=0), weights[:1])
    np.testing.assert_array_equal(
        get_subunits(weights, 3, 3, threshold_morans_i=0, mask=np.ones((3, 3), bool)),
        weights[:1],
    )


def test_a_node_with_no_weight_is_never_a_subunit():
    weights = np.zeros((1, 9))
    assert np.isnan(morans_i(weights[0].reshape(3, 3)))
    assert get_subunits(weights, 3, 3).shape == (0, 9)


def test_subunit_sizes_keep_one_entry_per_subunit():
    x, y = np.indices((20, 20))
    blob = np.exp(-((x - 10) ** 2 + (y - 9) ** 2) / 8.0).ravel()
    sizes = get_subunit_sizes(np.stack([blob, -blob]), 20, 20)
    assert sizes.shape == (2,)
    assert sizes[0] > 0 and np.isnan(sizes[1])
    assert get_subunit_sizes(np.stack([blob]), 20, 20, pixel_size_um=60)[0] == pytest.approx(
        2 * sizes[0])
    assert get_rf_size(blob.reshape(20, 20)) == pytest.approx(sizes[0])
