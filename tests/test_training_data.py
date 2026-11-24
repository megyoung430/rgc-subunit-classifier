from pathlib import Path

import h5py
import pytest
import numpy as np

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
    examples = gen_eses(
        2, 1, spikes, np.array([1.0, -1.0]), np.ones((2, 1)),
        (slice(0, 2), slice(0, 1)), np.array([[True], [False]]),
        2, 4, 1, tmp_path,
    )
    # A count of three still creates one autoencoder example, as in the notebook.
    assert examples.shape == (1, 1)
    np.testing.assert_allclose(examples[0], [-0.5])


def test_sta_cache_is_reused_only_for_identical_spikes(tmp_path):
    stim_dir, cache = tmp_path / "stim", tmp_path / "cache"
    stim_dir.mkdir()
    rng = np.random.default_rng(0)
    for trial in range(2):
        save_stimulus(stim_dir, trial, rng.choice([-1, 1], size=(2, 6)))
    spikes = np.array([[0, 0], [0, 0], [1, 0], [0, 2], [1, 1], [0, 1]])

    first = gen_sta(2, 1, spikes, 0, 2, 6, 2, stim_path=stim_dir, sta_path=cache)
    assert len(list(cache.glob("*.h5"))) == 1
    # A cache hit must not need the stimulus files at all.
    for path in stim_dir.iterdir():
        path.unlink()
    again = gen_sta(2, 1, spikes, 0, 2, 6, 2, stim_path=stim_dir, sta_path=cache)
    for a, b in zip(first, again):
        np.testing.assert_array_equal(a, b)

    # Fewer trials means different spikes, so the cached STA must not be used.
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
