from pathlib import Path

import h5py
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
