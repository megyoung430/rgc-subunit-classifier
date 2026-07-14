import numpy as np

from src.analysis.analyze_subunits import get_morans_i, get_subunits
from src.analysis.fitellipse import fitellipse


def test_fitellipse_recovers_gaussian_center():
    x, y = np.indices((21, 17))
    image = np.exp(-((x - 10) ** 2 / 8 + (y - 7) ** 2 / 4))
    center_x, center_y, major, minor, _ = fitellipse(image, raw=True)
    np.testing.assert_allclose([center_x, center_y], [10, 7], atol=0.1)
    assert major >= minor > 0


def test_subunit_filter_uses_morans_threshold():
    smooth = np.array([[0, 0, 0], [0, 1, 1], [0, 1, 1]], dtype=float)
    checker = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=float)
    assert get_morans_i(smooth, 3, 3) > get_morans_i(checker, 3, 3)
    weights = np.stack([smooth.ravel(), checker.ravel()])
    selected = get_subunits(weights, 3, 3, threshold_morans_i=0)
    np.testing.assert_array_equal(selected, weights[:1])
