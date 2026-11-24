"""Fit and evaluate the conventional linear-nonlinear STA model."""

from pathlib import Path

import h5py
import numpy as np
from scipy import optimize

from ..data_cache import check_stimuli, cropped_frozen_stimulus, load_spike_data


# Steep threshold curves (a2 around 5) need far more evaluations than
# leastsq's default of 800 to converge.
MAX_FIT_EVALUATIONS = 20000


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


def _trial(stimuli, stim_path, trial, x, y, num_frames, crop):
    if stimuli is not None:
        return np.asarray(stimuli[trial])
    return _load_trial(stim_path, trial, x, y, num_frames, crop)


def _frozen_inputs(cell_num, crop, data_path, stim_frz_path, cache_dir):
    """Frozen-noise spike counts for one cell and the cropped frozen stimulus."""
    spikes_dir = None if cache_dir is None else Path(cache_dir) / "spikes"
    stimuli_dir = None if cache_dir is None else Path(cache_dir) / "stimuli"
    spikecounts = load_spike_data(data_path, spikes_dir, keys=("spk2",))["spk2"][cell_num]
    stimulus = cropped_frozen_stimulus(stim_frz_path, crop, stimuli_dir)
    if stimulus.shape[1] != spikecounts.shape[0]:
        raise ValueError(
            f"frozen stimulus has {stimulus.shape[1]} frames but spk2 has "
            f"{spikecounts.shape[0]}"
        )
    return spikecounts, stimulus


def separable_ln_filter(spatial_sta, temporal_sta, crop, mask):
    """Masked spatial STA times temporal STA, shaped like the cropped full STA.

    This LN filter has the same structure as the subunit models (spatial maps
    inside the ellipse, one shared temporal filter), so comparing the two
    isolates the effect of the subunit nonlinearity. The full cropped STA has
    an independent, noisy weight for every pixel and lag, and predicts held-out
    responses worse for that reason alone.

    ``gen_sta`` cuts both slices through the STA's largest absolute value, so
    each carries that value's sign and their product is always positive
    there. For an OFF cell the product is therefore the negative of the STA;
    multiplying by the sign of the spatial STA at its peak restores it.

    Returns:
        Crop pixels (C order) × lags, the layout ``gen_sta_model`` expects.
    """
    spatial_sta = np.asarray(spatial_sta)
    polarity = np.sign(spatial_sta.flat[np.abs(spatial_sta).argmax()])
    spatial = spatial_sta[crop]
    mask = np.asarray(mask, dtype=bool)
    if mask.shape != spatial.shape:
        raise ValueError(f"mask has shape {mask.shape}; the crop has shape {spatial.shape}")
    return polarity * np.outer(np.where(mask, spatial, 0.0).ravel(), np.asarray(temporal_sta))


def fit_softplus(centers, rates):
    """Least-squares fit of :func:`nonlin` to binned firing rates.

    Raises:
        RuntimeError: If the fit does not converge, rather than returning
            parameters from wherever the optimizer stopped.
    """
    centers, rates = np.asarray(centers, dtype=float), np.asarray(rates, dtype=float)
    if not (np.all(np.isfinite(centers)) and np.all(np.isfinite(rates))):
        raise ValueError("binned drives and rates must be finite")
    # A fixed starting point fails for drives or rates far from 1, and one
    # on the data's scale can stall in a poor solution, so try both and keep
    # the converged fit with the smallest error.
    spread = centers.std() or 1.0
    starts = ([1.0, 1.0, 1.0], [max(rates.max(), 1e-6), 1 / spread, -centers.mean()])
    fits, messages = [], []
    for start in starts:
        parameters, _, info, message, status = optimize.leastsq(
            lambda p, x_values, y_values: y_values - nonlin(x_values, *p),
            start, args=(centers, rates), full_output=True, maxfev=MAX_FIT_EVALUATIONS,
        )
        if status in (1, 2, 3, 4) and np.all(np.isfinite(parameters)):
            fits.append((float(np.sum(info["fvec"] ** 2)), parameters))
        messages.append(message)
    if not fits:
        raise RuntimeError(f"the softplus fit did not converge: {'; '.join(messages)}")
    return min(fits, key=lambda fit: fit[0])[1]


def gen_sta_model(
    x, y, spikecounts, crop, crop_x, crop_y, filter_length, num_frames,
    num_trials, stac, stim_path, dt=1 / 30, stimuli=None,
):
    """Fit the STA model's scalar output nonlinearity on training trials.

    ``stimuli`` is an optional preloaded cropped stimulus of shape
    ``(trials, crop pixels, num_frames)``; when given, no files are read.
    """
    if stimuli is not None:
        check_stimuli(stimuli, crop_x * crop_y, num_frames, num_trials)
    filtered = []
    for trial in range(num_trials):
        stimulus = _trial(stimuli, stim_path, trial, x, y, num_frames, crop)
        for frame in range(filter_length, num_frames):
            frames = stimulus[:, frame-filter_length:frame]
            filtered.append(np.asarray(stac).flatten("F").dot(frames.flatten("F")))
    counts = np.asarray(spikecounts)[filter_length:, :].T.ravel()
    centers, rates = _binned_response(np.asarray(filtered), counts, dt)
    if len(centers) < 3:
        raise ValueError("not enough populated response bins to fit the STA model")
    return fit_softplus(centers, rates)


def get_sta_predictions(
    cell_num, params_sta, stac, crop, crop_x, crop_y, data_path,
    stim_frz_path, dt=1 / 30, filter_length=20, cache_dir=None,
):
    """Predict the frozen-noise response and return its Pearson correlation.

    With ``cache_dir``, spike counts and the cropped frozen stimulus are read
    through the caches in :mod:`src.data_cache` (subfolders ``spikes`` and
    ``stimuli``).
    """
    spikecounts, stimulus = _frozen_inputs(
        cell_num, crop, data_path, stim_frz_path, cache_dir
    )
    num_frames = spikecounts.shape[0]
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
