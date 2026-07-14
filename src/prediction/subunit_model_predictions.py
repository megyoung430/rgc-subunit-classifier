"""Fit and evaluate a rectified subunit model."""

from pathlib import Path

import h5py
import numpy as np
from scipy import io as spio, optimize

from .sta_model_predictions import _binned_response, _load_trial, nonlin


def gen_subunit_model(
    subunits, x, y, spikecounts, filter_length, num_frames, num_trials,
    spatial_sta, temp_sta, crop, crop_x, crop_y, stim_path, dt=1 / 30,
):
    """Fit spatial combination weights and the scalar output nonlinearity."""
    subunits = np.asarray(subunits, dtype=float).reshape(len(subunits), -1).copy()
    norms = np.linalg.norm(subunits, axis=1)
    if np.any(norms == 0):
        raise ValueError("subunits must have non-zero norms")
    subunits /= norms[:, None]
    target = np.asarray(spatial_sta)[crop].ravel()
    if target[np.abs(target).argmax()] < 0:
        target = -target
    weights = np.linalg.lstsq(subunits.T, target, rcond=None)[0]

    filtered = []
    for trial in range(num_trials):
        stimulus = _load_trial(stim_path, trial, x, y, num_frames, crop)
        for frame in range(filter_length, num_frames):
            frames = stimulus[:, frame-filter_length:frame] * np.asarray(temp_sta)
            activations = np.maximum(subunits.dot(frames.mean(axis=1)), 0)
            filtered.append(activations.dot(weights))
    counts = np.asarray(spikecounts)[filter_length:, :].T.ravel()
    centers, rates = _binned_response(np.asarray(filtered), counts, dt)
    params = optimize.leastsq(
        lambda p, x_values, y_values: y_values - nonlin(x_values, *p),
        [1, 1, 1], args=(centers, rates),
    )[0]
    return params, weights, subunits


def get_subunit_predictions(
    cell_num, params_subunits, weights, subunits, temp_sta, crop,
    crop_x, crop_y, data_path, stim_frz_path, dt=1 / 30, filter_length=20,
):
    """Predict the frozen-noise response and return its Pearson correlation."""
    spikecounts = spio.loadmat(data_path)["spk2"][cell_num]
    num_frames = spikecounts.shape[0]
    with h5py.File(Path(stim_frz_path) / "stim_frz.h5", "r") as handle:
        stimulus = handle["stim_frz"][:]
    stimulus = stimulus[crop]
    stimulus = np.moveaxis(stimulus, 0, 1).reshape(
        crop_y * crop_x, num_frames, order="F"
    )
    predictions = []
    for frame in range(filter_length, num_frames):
        frames = stimulus[:, frame-filter_length:frame] * np.asarray(temp_sta)
        activations = np.maximum(np.asarray(subunits).dot(frames.mean(axis=1)), 0)
        predictions.append(nonlin(activations.dot(weights), *params_subunits))
    predictions = np.asarray(predictions)
    actual = spikecounts.mean(axis=1) / dt
    correlation = np.corrcoef(actual[filter_length:], predictions)[0, 1]
    return predictions, correlation, actual, np.arange(num_frames) * dt


gen_subunit_model = gen_subunit_model
get_subunit_predictions = get_subunit_predictions
