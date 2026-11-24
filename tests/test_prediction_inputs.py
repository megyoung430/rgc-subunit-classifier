import h5py
import numpy as np
import pytest

from src.prediction.sta_model_predictions import gen_sta_model
from src.prediction.subunit_model_predictions import gen_subunit_model


def test_prediction_fits_accept_preloaded_cropped_stimuli(tmp_path):
    x, y, frames, trials, filt = 4, 4, 60, 3, 3
    rng = np.random.default_rng(3)
    stims = [rng.choice([-1, 1], size=(x * y, frames)) for _ in range(trials)]
    for trial, values in enumerate(stims):
        with h5py.File(tmp_path / f"stim{trial:04d}.h5", "w") as handle:
            handle["stim"] = values
    crop = (slice(0, 2), slice(0, 2))
    preloaded = np.stack(
        [s.reshape(x, y, frames)[crop].reshape(-1, frames) for s in stims]
    )
    stac = rng.normal(size=(4, filt))
    # Spikes follow the cropped stimulus one frame earlier, so the softplus fits converge.
    drive = np.stack([np.roll(p.sum(axis=0), 1) for p in preloaded], axis=1)
    spikes = rng.poisson(np.log1p(np.exp(drive)))

    on_disk = gen_sta_model(x, y, spikes, crop, 2, 2, filt, frames, trials, stac, tmp_path)
    in_memory = gen_sta_model(x, y, spikes, crop, 2, 2, filt, frames, trials, stac, None,
                              stimuli=preloaded)
    np.testing.assert_allclose(on_disk, in_memory)

    subunits = np.eye(4)[:2]
    spatial = rng.normal(size=(x, y))
    temporal = rng.normal(size=filt)
    a = gen_subunit_model(subunits, x, y, spikes, filt, frames, trials, spatial, temporal,
                          crop, 2, 2, tmp_path)
    b = gen_subunit_model(subunits, x, y, spikes, filt, frames, trials, spatial, temporal,
                          crop, 2, 2, None, stimuli=preloaded)
    for left, right in zip(a, b):
        np.testing.assert_allclose(left, right)
