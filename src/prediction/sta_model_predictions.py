"""Fit and evaluate the conventional linear-nonlinear STA model."""

from pathlib import Path

import h5py
import numpy as np
from scipy import io as spio, optimize


def nonlin(x, a1, a2, a3):
    """Softplus output nonlinearity used by the original notebook."""
    return a1 * np.logaddexp(0, a2 * (np.asarray(x) + a3))


def _binned_response(filtered, counts, dt, num_bins=40):
    edges = np.quantile(filtered, np.linspace(0, 1, num_bins + 1)[1:-1])
    groups = np.digitize(filtered, edges)
    centers, rates = [], []
    for group in range(num_bins):
        selected = groups == group
        if selected.any():
            centers.append(filtered[selected].mean())
            rates.append(counts[selected].mean() / dt)
    return np.asarray(centers), np.asarray(rates)


def _load_trial(stim_path, trial, x, y, num_frames, crop):
    with h5py.File(Path(stim_path) / f"stim{trial:04d}.h5", "r") as handle:
        stimulus = handle["stim"][:]
    return stimulus.reshape(x, y, num_frames)[crop].reshape(-1, num_frames)


def gen_sta_model(
    x, y, spikecounts, crop, crop_x, crop_y, filter_length, num_frames,
    num_trials, stac, stim_path, dt=1 / 30,
):
    """Fit the STA model's scalar output nonlinearity on training trials."""
    filtered = []
    for trial in range(num_trials):
        stimulus = _load_trial(stim_path, trial, x, y, num_frames, crop)
        for frame in range(filter_length, num_frames):
            frames = stimulus[:, frame-filter_length:frame]
            filtered.append(np.asarray(stac).flatten("F").dot(frames.flatten("F")))
    counts = np.asarray(spikecounts)[filter_length:, :].T.ravel()
    centers, rates = _binned_response(np.asarray(filtered), counts, dt)
    if len(centers) < 3:
        raise ValueError("not enough populated response bins to fit the STA model")
    return optimize.leastsq(
        lambda p, x_values, y_values: y_values - nonlin(x_values, *p),
        [1, 1, 1], args=(centers, rates),
    )[0]


def get_sta_predictions(
    cell_num, params_sta, stac, crop, crop_x, crop_y, data_path,
    stim_frz_path, dt=1 / 30, filter_length=20,
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
        frames = stimulus[:, frame-filter_length:frame]
        value = np.asarray(stac).flatten("F").dot(frames.flatten("F"))
        predictions.append(nonlin(value, *params_sta))
    predictions = np.asarray(predictions)
    actual = spikecounts.mean(axis=1) / dt
    correlation = np.corrcoef(actual[filter_length:], predictions)[0, 1]
    return predictions, correlation, actual, np.arange(num_frames) * dt


gen_STA_model = gen_sta_model
get_STA_predictions = get_sta_predictions
