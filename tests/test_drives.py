import numpy as np

from src.prediction.sta_model_predictions import ln_drive
from src.prediction.subunit_model_predictions import subunit_drive


def loop_ln(stimulus, stac, filter_length):
    out = []
    for frame in range(filter_length, stimulus.shape[1]):
        frames = stimulus[:, frame - filter_length:frame]
        out.append(np.asarray(stac).flatten("F").dot(frames.flatten("F")))
    return np.array(out)


def loop_subunit(stimulus, subunits, weights, temp_sta, filter_length):
    out = []
    for frame in range(filter_length, stimulus.shape[1]):
        frames = stimulus[:, frame - filter_length:frame] * temp_sta
        out.append(np.maximum(subunits.dot(frames.mean(axis=1)), 0).dot(weights))
    return np.array(out)


def test_vectorized_drives_match_the_frame_by_frame_loops():
    rng = np.random.default_rng(0)
    stimulus = rng.choice([-1, 1], size=(12, 80)).astype(np.int8)
    stac, temp = rng.normal(size=(12, 5)), rng.normal(size=5)
    subunits, weights = rng.normal(size=(3, 12)), rng.normal(size=3)
    np.testing.assert_allclose(ln_drive(stimulus, stac, 5), loop_ln(stimulus, stac, 5))
    np.testing.assert_allclose(subunit_drive(stimulus, subunits, weights, temp, 5),
                               loop_subunit(stimulus, subunits, weights, temp, 5))
