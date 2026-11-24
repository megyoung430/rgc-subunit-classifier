from pathlib import Path

import h5py
import numpy as np
import pytest

import src.simulation.generate_training_data as training_data
from src.simulation.generate_training_data import gen_eses, gen_ses, gen_sta


def save_stimulus(path: Path, trial: int, values):
    with h5py.File(path / f"stim{trial:04d}.h5", "w") as handle:
        handle["stim"] = np.asarray(values)


def test_gen_sta_is_spike_weighted_average(tmp_path):
    # Two pixels, five frames, one trial. A double spike at frame 4 selects
    # frames 2:4; normalization by two leaves that preceding stimulus intact.
    stimulus = np.array([[1, -1, 1, -1, 1], [-1, 1, -1, 1, -1]])
    save_stimulus(tmp_path, 0, stimulus)
    spikes = np.array([[0], [0], [0], [0], [2]])
    sta, spatial, temporal = gen_sta(
        2, 1, spikes, 0, 2, 5, 1, stim_path=tmp_path
    )
    np.testing.assert_array_equal(sta, stimulus[:, 2:4])
    assert np.isclose(np.linalg.norm(spatial), 1)
    assert np.isclose(np.linalg.norm(temporal), 1)


def test_gen_ses_applies_temporal_filter_mask_and_binary_labels(tmp_path):
    stimulus = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
    save_stimulus(tmp_path, 0, stimulus)
    spikes = np.array([[0], [0], [2], [0]])
    examples, labels = gen_ses(
        2, 1, spikes, np.array([1.0, -1.0]), np.ones((2, 1)),
        (slice(0, 2), slice(0, 1)), np.array([[True], [False]]),
        2, 4, 1, stim_path=tmp_path,
    )
    np.testing.assert_allclose(examples[:, 0], [-0.5, -0.5])
    np.testing.assert_array_equal(labels, [1, 0])


def test_gen_eses_keeps_only_spike_frames_once(tmp_path):
    stimulus = np.array([[1, 2, 3, 4], [10, 20, 30, 40]])
    save_stimulus(tmp_path, 0, stimulus)
    spikes = np.array([[0], [0], [3], [0]])
    examples, trials = gen_eses(
        2, 1, spikes, np.array([1.0, -1.0]), np.ones((2, 1)),
        (slice(0, 2), slice(0, 1)), np.array([[True], [False]]),
        2, 4, 1, tmp_path,
    )
    # A count of three still creates one autoencoder example, as in the notebook.
    assert examples.shape == (1, 1)
    np.testing.assert_array_equal(trials, [0])
    np.testing.assert_allclose(examples[0], [-0.5])


def test_sta_cache_is_reused_only_for_identical_spikes(tmp_path, monkeypatch):
    stim_dir, cache = tmp_path / "stim", tmp_path / "cache"
    stim_dir.mkdir()
    rng = np.random.default_rng(0)
    for trial in range(2):
        save_stimulus(stim_dir, trial, rng.choice([-1, 1], size=(2, 6)))
    spikes = np.array([[0, 0], [0, 0], [1, 0], [0, 2], [1, 1], [0, 1]])

    first = gen_sta(2, 1, spikes, 0, 2, 6, 2, stim_path=stim_dir, sta_path=cache)
    assert len(list(cache.glob("*.h5"))) == 1
    # A cache hit only checks the stimulus files' sizes and times; it reads none.
    def no_reading(*args):
        raise AssertionError("stimulus file read on a cache hit")

    monkeypatch.setattr(training_data, "_load_stimulus", no_reading)
    again = gen_sta(2, 1, spikes, 0, 2, 6, 2, stim_path=stim_dir, sta_path=cache)
    monkeypatch.undo()
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)

    # Fewer trials means different spikes, so the cached STA must not be used.
    (stim_dir / "stim0000.h5").unlink()
    save_stimulus(stim_dir, 0, np.ones((2, 6)))
    subset = gen_sta(2, 1, spikes[:, :1], 0, 2, 6, 1, stim_path=stim_dir, sta_path=cache)
    assert len(list(cache.glob("*.h5"))) == 2
    np.testing.assert_array_equal(subset[0], np.ones((2, 2)))


def test_sta_cache_cannot_be_written_into_the_archive(tmp_path, monkeypatch):
    import src.outputs

    archive = tmp_path / "results" / "colab_legacy"
    monkeypatch.setattr(src.outputs, "ARCHIVE", archive)
    save_stimulus(tmp_path, 0, np.array([[1, -1, 1], [-1, 1, -1]]))
    with pytest.raises(PermissionError):
        gen_sta(2, 1, np.array([[0], [0], [1]]), 0, 2, 3, 1,
                stim_path=tmp_path, sta_path=archive / "sta")


def test_sta_cache_distinguishes_stimulus_folders(tmp_path):
    cache = tmp_path / "cache"
    real, flipped = tmp_path / "real", tmp_path / "flipped"
    real.mkdir()
    flipped.mkdir()
    stimulus = np.array([[1, -1, 1, -1], [-1, 1, -1, 1]])
    save_stimulus(real, 0, stimulus)
    save_stimulus(flipped, 0, -stimulus)
    spikes = np.array([[0], [0], [1], [0]])
    a = gen_sta(2, 1, spikes, 0, 2, 4, 1, stim_path=real, sta_path=cache)[0]
    b = gen_sta(2, 1, spikes, 0, 2, 4, 1, stim_path=flipped, sta_path=cache)[0]
    np.testing.assert_array_equal(b, -a)
    assert len(list(cache.glob("*.h5"))) == 2


def test_corrupt_sta_cache_file_is_named_in_the_error(tmp_path):
    save_stimulus(tmp_path, 0, np.array([[1, -1, 1], [-1, 1, -1]]))
    spikes = np.array([[0], [0], [1]])
    cache = tmp_path / "cache"
    gen_sta(2, 1, spikes, 0, 2, 3, 1, stim_path=tmp_path, sta_path=cache)
    (path,) = cache.glob("*.h5")
    path.write_bytes(path.read_bytes()[:100])
    with pytest.raises(OSError, match="delete it"):
        gen_sta(2, 1, spikes, 0, 2, 3, 1, stim_path=tmp_path, sta_path=cache)


def test_sta_cache_cannot_be_combined_with_generated_stimuli(tmp_path):
    with pytest.raises(ValueError, match="sta_path"):
        gen_sta(2, 1, np.zeros((3, 1)), 0, 2, 3, 1, sta_path=tmp_path,
                generate_stimuli=True, running_seed=0,
                stimulus_generator=lambda seed, n: (np.ones(n), seed))


def test_ensembles_from_preloaded_stimuli_match_file_loading(tmp_path):
    x, y, frames, trials = 3, 2, 6, 2
    rng = np.random.default_rng(2)
    stims = [rng.choice([-1, 1], size=(x * y, frames)) for _ in range(trials)]
    for trial, values in enumerate(stims):
        save_stimulus(tmp_path, trial, values)
    crop = (slice(0, 2), slice(0, 2))
    mask = np.array([[True, True], [False, True]])
    temporal = np.array([0.5, -1.0])
    spikes = rng.integers(0, 2, size=(frames, trials))
    preloaded = np.stack(
        [s.reshape(x, y, frames)[crop].reshape(-1, frames) for s in stims]
    )
    args = (x, y, spikes, temporal, np.ones((2, 2)), crop, mask, 2, frames, trials)

    from_files = gen_ses(*args, stim_path=tmp_path)
    from_array = gen_ses(*args, stimuli=preloaded)
    for a, b in zip(from_files, from_array):
        np.testing.assert_array_equal(a, b)
    for a, b in zip(gen_eses(*args, tmp_path), gen_eses(*args, None, stimuli=preloaded)):
        np.testing.assert_array_equal(a, b)


def test_preloaded_stimuli_with_the_wrong_crop_are_rejected(tmp_path):
    spikes = np.zeros((4, 1), dtype=int)
    args = (2, 2, spikes, np.array([1.0]), np.ones((2, 2)),
            (slice(0, 2), slice(0, 2)), np.ones((2, 2), dtype=bool), 1, 4, 1)
    with pytest.raises(ValueError, match="pixels"):
        gen_ses(*args, stimuli=np.zeros((1, 9, 4)))
    with pytest.raises(ValueError, match="generate_stimuli"):
        gen_ses(*args, stimuli=np.zeros((1, 4, 4)), generate_stimuli=True,
                running_seed=0, stimulus_generator=lambda s, n: (np.ones(n), s))


def test_sta_cache_written_by_another_run_meanwhile_is_not_an_error(tmp_path, monkeypatch):
    def already_written(path, write):
        raise FileExistsError(path)

    monkeypatch.setattr(training_data, "write_new_file", already_written)
    save_stimulus(tmp_path, 0, np.array([[1, -1, 1], [-1, 1, -1]]))
    sta, _, _ = gen_sta(2, 1, np.array([[0], [0], [1]]), 0, 2, 3, 1,
                        stim_path=tmp_path, sta_path=tmp_path / "cache")
    np.testing.assert_array_equal(sta, [[1, -1], [-1, 1]])
