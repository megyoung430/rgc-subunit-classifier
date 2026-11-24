import numpy as np
import pytest

from src.analysis.fitellipse import fitellipse
from src.simulation.generate_training_data import crop_sta


def gaussian(shape, centre, widths, angle=0.0):
    x, y = np.indices(shape, dtype=float)
    dx, dy = x - centre[0], y - centre[1]
    c, s = np.cos(angle), np.sin(angle)
    u, v = c * dx + s * dy, -s * dx + c * dy
    return np.exp(-(u**2 / (2 * widths[0] ** 2) + v**2 / (2 * widths[1] ** 2)))


@pytest.mark.parametrize("angle", [0.0, 0.6, np.pi / 2])
def test_mask_fills_the_ellipse_with_no_holes(angle):
    image = gaussian((60, 50), (30.3, 24.6), (5.0, 2.5), angle)
    sigma = 3
    cropped, crop, mask = crop_sta(image, sigma)
    _, _, d_major, d_minor, _ = fitellipse(image, sigma=sigma, raw=True)
    area = np.pi * d_major * d_minor / 4
    assert abs(mask.sum() - area) / area < 0.05
    for line in (*mask.T, *mask):  # every row and column is one unbroken run
        filled = np.flatnonzero(line)
        if filled.size:
            assert filled[-1] - filled[0] + 1 == filled.size
    # The crop is the mask's bounding box, so no edge row or column is empty.
    assert mask[0].any() and mask[-1].any() and mask[:, 0].any() and mask[:, -1].any()
    np.testing.assert_array_equal(cropped[~mask], 0)
    np.testing.assert_array_equal(cropped[mask], image[crop][mask])


def test_peak_pixel_is_inside_the_mask():
    image = gaussian((40, 40), (20, 20), (4.0, 3.0))
    _, crop, mask = crop_sta(image, 3)
    peak = np.unravel_index(image.argmax(), image.shape)
    assert mask[peak[0] - crop[0].start, peak[1] - crop[1].start]


def test_ellipse_running_off_the_frame_is_clipped():
    image = gaussian((30, 30), (2.0, 15.0), (4.0, 3.0))
    _, crop, mask = crop_sta(image, 3)
    assert crop[0].start == 0
    assert mask[0].any()


def test_negative_sta_is_cropped_like_a_positive_one():
    image = gaussian((40, 40), (18.0, 22.0), (4.0, 2.0), 0.3)
    _, crop_pos, mask_pos = crop_sta(image, 3)
    _, crop_neg, mask_neg = crop_sta(-image, 3)
    assert crop_pos == crop_neg
    np.testing.assert_array_equal(mask_pos, mask_neg)
