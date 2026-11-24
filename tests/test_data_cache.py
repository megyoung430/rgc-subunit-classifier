import h5py
import numpy as np
import pytest
from scipy.io import savemat

from src.data_cache import (
    check_stimuli,
    cropped_frozen_stimulus,
    cropped_stimulus,
    load_spike_data,
)


def write_trials(folder, trials):
    folder.mkdir(exist_ok=True)
    for index, values in enumerate(trials):
        with h5py.File(folder / f"stim{index:04d}.h5", "w") as handle:
            handle["stim"] = values


def expected_crop(trials, x, y, frames, crop, n=None):
    return np.stack(
        [t.reshape(x, y, frames)[crop].reshape(-1, frames) for t in trials[:n]]
    )


def test_spike_data_cache_round_trips_and_follows_file_content(tmp_path):
    mat = tmp_path / "cells.mat"
    spk1 = np.arange(24).reshape(2, 4, 3)
    savemat(mat, {"spk1": spk1, "spk2": spk1[:, :, :2]})
    cache = tmp_path / "cache"

    first = load_spike_data(mat, cache_dir=cache)
    np.testing.assert_array_equal(first["spk1"], spk1)
    np.testing.assert_array_equal(first["spk2"], spk1[:, :, :2])
    assert len(list(cache.glob("*.npz"))) == 1
    np.testing.assert_array_equal(load_spike_data(mat, cache_dir=cache)["spk1"], spk1)

    savemat(mat, {"spk1": spk1 + 1, "spk2": spk1[:, :, :2]})
    np.testing.assert_array_equal(load_spike_data(mat, cache_dir=cache)["spk1"], spk1 + 1)
    assert len(list(cache.glob("*.npz"))) == 2


def test_spike_data_keys_are_optional_and_each_caller_gets_its_own_dict(tmp_path):
    mat = tmp_path / "cells.mat"
    savemat(mat, {"spk1": np.ones((1, 2, 2))})
    spikes = load_spike_data(mat, keys=("spk1",))
    assert set(spikes) == {"spk1"}
    spikes["spk1"] = None
    assert load_spike_data(mat, keys=("spk1",))["spk1"].shape == (1, 2, 2)
    with pytest.raises(ValueError, match="spk2"):
        load_spike_data(mat, keys=("spk2",))


def test_cropped_stimulus_is_trial_first_cached_and_reused(tmp_path, monkeypatch):
    import src.data_cache as data_cache

    x, y, frames = 4, 3, 5
    rng = np.random.default_rng(1)
    trials = [rng.choice([-1, 1], size=(x * y, frames)).astype(np.int8) for _ in range(3)]
    write_trials(tmp_path / "stim", trials)
    crop = (slice(1, 3), slice(0, 2))
    cache = tmp_path / "cache"

    cropped = cropped_stimulus(tmp_path / "stim", crop, x, y, frames, 3, cache_dir=cache)
    assert cropped.shape == (3, 4, frames)
    np.testing.assert_array_equal(cropped, expected_crop(trials, x, y, frames, crop))
    assert len(list(cache.glob("*.npy"))) == 1

    def no_reading(*args):
        raise AssertionError("stimulus files read on a cache hit")

    monkeypatch.setattr(data_cache, "_read_cropped_trials", no_reading)
    again = cropped_stimulus(tmp_path / "stim", crop, x, y, frames, 3, cache_dir=cache)
    np.testing.assert_array_equal(again, cropped)
    # A smaller prefix of trials is served from the existing file.
    prefix = cropped_stimulus(tmp_path / "stim", crop, x, y, frames, 2, cache_dir=cache)
    np.testing.assert_array_equal(prefix, expected_crop(trials, x, y, frames, crop, 2))
    assert len(list(cache.glob("*.npy"))) == 1


def test_cropped_stimulus_cache_distinguishes_stimulus_folders(tmp_path):
    x, y, frames = 2, 2, 3
    real = [np.array([[1, -1, 1]] * 4, dtype=np.int8)]
    write_trials(tmp_path / "real", real)
    write_trials(tmp_path / "flipped", [-real[0]])
    crop = (slice(0, 2), slice(0, 2))
    cache = tmp_path / "cache"
    a = cropped_stimulus(tmp_path / "real", crop, x, y, frames, 1, cache_dir=cache)
    b = cropped_stimulus(tmp_path / "flipped", crop, x, y, frames, 1, cache_dir=cache)
    np.testing.assert_array_equal(b, -np.asarray(a))


def test_cropped_stimulus_keeps_the_file_dtype(tmp_path):
    values = np.full((4, 3), 0.6)
    write_trials(tmp_path / "stim", [values])
    cropped = cropped_stimulus(tmp_path / "stim", (slice(0, 2), slice(0, 2)), 2, 2, 3, 1)
    assert cropped.dtype == values.dtype
    np.testing.assert_allclose(cropped, 0.6)


def test_missing_stimulus_file_is_reported(tmp_path):
    write_trials(tmp_path / "stim", [np.ones((4, 2), dtype=np.int8)])
    with pytest.raises(FileNotFoundError):
        cropped_stimulus(tmp_path / "stim", (slice(0, 2), slice(0, 2)), 2, 2, 2, 2,
                         cache_dir=tmp_path / "cache")


def test_cropped_stimulus_cache_refuses_the_archive(tmp_path, monkeypatch):
    import src.outputs

    archive = tmp_path / "results" / "colab_legacy"
    monkeypatch.setattr(src.outputs, "ARCHIVE", archive)
    write_trials(tmp_path / "stim", [np.ones((4, 2), dtype=np.int8)])
    with pytest.raises(PermissionError):
        cropped_stimulus(tmp_path / "stim", (slice(0, 1), slice(0, 2)), 2, 2, 2, 1,
                         cache_dir=archive / "cache")


def test_cropped_frozen_stimulus_matches_prediction_layout_and_file(tmp_path):
    x, y, frames = 4, 3, 6
    frozen = np.arange(x * y * frames).reshape(x, y, frames)
    for name, values in (("a", frozen), ("b", -frozen)):
        (tmp_path / name).mkdir()
        with h5py.File(tmp_path / name / "stim_frz.h5", "w") as handle:
            handle["stim_frz"] = values
    crop = (slice(1, 4), slice(1, 3))
    cache = tmp_path / "cache"
    cropped = cropped_frozen_stimulus(tmp_path / "a", crop, cache_dir=cache)
    # The layout the prediction code builds: pixels in C order, frames last.
    legacy = np.moveaxis(frozen[crop], 0, 1).reshape(2 * 3, frames, order="F")
    np.testing.assert_array_equal(cropped, legacy)
    other = cropped_frozen_stimulus(tmp_path / "b", crop, cache_dir=cache)
    np.testing.assert_array_equal(other, -legacy)


def test_check_stimuli_rejects_mismatched_arrays():
    stimuli = np.zeros((3, 4, 10))
    check_stimuli(stimuli, n_pixels=4, num_frames=10, num_trials=2)
    with pytest.raises(ValueError, match="pixels"):
        check_stimuli(stimuli, n_pixels=6, num_frames=10, num_trials=2)
    with pytest.raises(ValueError, match="trials"):
        check_stimuli(stimuli, n_pixels=4, num_frames=10, num_trials=5)


def test_smallest_cached_file_with_enough_trials_is_used(tmp_path):
    x, y, frames = 2, 2, 3
    rng = np.random.default_rng(4)
    trials = [rng.choice([-1, 1], size=(4, frames)).astype(np.int8) for _ in range(5)]
    write_trials(tmp_path / "stim", trials)
    crop, cache = (slice(0, 2), slice(0, 2)), tmp_path / "cache"
    cropped_stimulus(tmp_path / "stim", crop, x, y, frames, 3, cache_dir=cache)
    cropped_stimulus(tmp_path / "stim", crop, x, y, frames, 5, cache_dir=cache)
    two = cropped_stimulus(tmp_path / "stim", crop, x, y, frames, 2, cache_dir=cache)
    assert str(two.filename).endswith(" 3 trials.npy")
    np.testing.assert_array_equal(two, expected_crop(trials, x, y, frames, crop, 2))
    assert sorted(p.name.split()[-2] for p in cache.glob("stimulus *.npy")) == ["3", "5"]


def test_cache_written_by_another_run_meanwhile_is_used(tmp_path, monkeypatch):
    import src.data_cache as data_cache

    write_trials(tmp_path / "stim", [np.ones((4, 2), dtype=np.int8)])
    crop, cache = (slice(0, 2), slice(0, 2)), tmp_path / "cache"
    real_read = data_cache._read_cropped_trials

    def other_run_finishes_first(*args):
        values = real_read(*args)
        monkeypatch.setattr(data_cache, "_read_cropped_trials", real_read)
        cropped_stimulus(tmp_path / "stim", crop, 2, 2, 2, 1, cache_dir=cache)
        return values

    monkeypatch.setattr(data_cache, "_read_cropped_trials", other_run_finishes_first)
    result = cropped_stimulus(tmp_path / "stim", crop, 2, 2, 2, 1, cache_dir=cache)
    np.testing.assert_array_equal(result, np.ones((1, 4, 2)))
    assert len(list(cache.glob("stimulus *.npy"))) == 1
