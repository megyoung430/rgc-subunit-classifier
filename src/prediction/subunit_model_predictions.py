"""Fit and evaluate a rectified subunit model."""


import numpy as np

from ..data_cache import check_stimuli
from .sta_model_predictions import _binned_response, _frozen_inputs, _trial, fit_softplus, nonlin


def subunit_drive(stimulus, subunits, weights, temp_sta, filter_length):
    """Drive of the rectified subunit model for one trial.

    Each preceding window of ``filter_length`` frames is weighted by the
    temporal STA and averaged over time, giving one value per pixel; each
    subunit's response to that is rectified (negative values set to 0), and
    the rectified responses are summed with ``weights``.

    Args:
        stimulus: Cropped stimulus, pixels × frames.
        subunits: Subunits × pixels spatial filters.
        weights: One combination weight per subunit.
        temp_sta: Temporal STA, ``filter_length`` values.
        filter_length: Number of past frames used.

    Returns:
        The drive for frames ``filter_length`` to the end.
    """
    windows = np.lib.stride_tricks.sliding_window_view(
        np.asarray(stimulus, dtype=float), filter_length, axis=1
    )[:, :-1]
    filtered = (windows * np.asarray(temp_sta, dtype=float)).mean(axis=2)
    activations = np.maximum(np.asarray(subunits, dtype=float) @ filtered, 0)
    return np.asarray(weights, dtype=float) @ activations


def gen_subunit_model(
    subunits, x, y, spikecounts, filter_length, num_frames, num_trials,
    spatial_sta, temp_sta, crop, crop_x, crop_y, stim_path, dt=1 / 30,
    stimuli=None,
):
    """Fit spatial combination weights and the scalar output nonlinearity.

    ``stimuli`` is an optional preloaded cropped stimulus of shape
    ``(trials, crop pixels, num_frames)``; when given, no files are read.
    """
    if stimuli is not None:
        check_stimuli(stimuli, crop_x * crop_y, num_frames, num_trials)
    subunits = np.asarray(subunits, dtype=float).reshape(len(subunits), -1).copy()
    norms = np.linalg.norm(subunits, axis=1)
    if np.any(norms == 0):
        raise ValueError("subunits must have non-zero norms")
    subunits /= norms[:, None]
    target = np.asarray(spatial_sta)[crop].ravel()
    if target[np.abs(target).argmax()] < 0:
        target = -target
    weights = np.linalg.lstsq(subunits.T, target, rcond=None)[0]

    filtered = np.concatenate([
        subunit_drive(_trial(stimuli, stim_path, trial, x, y, num_frames, crop),
                      subunits, weights, temp_sta, filter_length)
        for trial in range(num_trials)
    ])
    counts = np.asarray(spikecounts)[filter_length:, :].T.ravel()
    centers, rates = _binned_response(filtered, counts, dt)
    params = fit_softplus(centers, rates)
    return params, weights, subunits


def get_subunit_predictions(
    cell_num, params_subunits, weights, subunits, temp_sta, crop,
    crop_x, crop_y, data_path, stim_frz_path, dt=1 / 30, filter_length=20,
    cache_dir=None,
):
    """Predict the frozen-noise response and return its Pearson correlation.

    ``cache_dir`` works as in :func:`get_sta_predictions`.
    """
    spikecounts, stimulus = _frozen_inputs(
        cell_num, crop, data_path, stim_frz_path, cache_dir
    )
    num_frames = spikecounts.shape[0]
    predictions = nonlin(
        subunit_drive(stimulus, subunits, weights, temp_sta, filter_length), *params_subunits
    )
    actual = spikecounts.mean(axis=1) / dt
    correlation = np.corrcoef(actual[filter_length:], predictions)[0, 1]
    return predictions, correlation, actual, np.arange(num_frames) * dt

