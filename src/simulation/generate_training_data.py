"""Build spike-triggered averages and neural-network training examples."""

from pathlib import Path
from typing import Callable

import h5py
import numpy as np

from ..analysis.fitellipse import fitellipse

StimulusGenerator = Callable[[int, int], tuple[np.ndarray, int]]


def _stimulus_file(path: str | Path, trial: int) -> Path:
    return Path(path) / f"stim{trial:04d}.h5"


def _load_stimulus(path: str | Path, trial: int) -> np.ndarray:
    with h5py.File(_stimulus_file(path, trial), "r") as handle:
        return handle["stim"][:]


def _make_stimulus(generator, seed, x, y, num_frames):
    if generator is None:
        raise ValueError("stimulus_generator is required when generate_stimuli=True")
    stimulus, next_seed = generator(seed, x * y * num_frames)
    stimulus = np.asarray(stimulus, dtype=np.int8).reshape(
        (x * y, num_frames), order="F"
    )
    stimulus[stimulus == 0] = -1
    return stimulus, next_seed


def gen_sta(
    x,
    y,
    spikecounts,
    cell_num,
    filter_length,
    num_frames,
    num_trials,
    sta_path=None,
    sta_filename=None,
    spatial_sta_filename=None,
    temp_sta_filename=None,
    generate_stimuli=False,
    stim_path=None,
    running_seed=None,
    stimulus_generator=None,
    save_stimuli=False,
    save_stim_path=None,
    save_sta=False,
    save_sta_path=None,
):
    """Calculate the full, spatial, and temporal spike-triggered averages.

    The stimulus preceding each spike is weighted by the spike count. The full
    STA is their sum divided by the total number of included spikes. Spatial
    and temporal components are slices through the largest absolute STA value,
    each normalized to unit Euclidean norm.
    """
    if sta_path is not None:
        base = Path(sta_path)
        with h5py.File(base / spatial_sta_filename, "r") as handle:
            spatial_sta = handle["spatial_STA"][:]
        with h5py.File(base / temp_sta_filename, "r") as handle:
            temp_sta = handle["temp_STA"][:]
        with h5py.File(base / sta_filename, "r") as handle:
            sta = handle["STA"][:]
        return sta, spatial_sta, temp_sta

    if not generate_stimuli and stim_path is None:
        raise ValueError("stim_path is required when generate_stimuli=False")
    if generate_stimuli and running_seed is None:
        raise ValueError("running_seed is required when generating stimuli")

    spikecounts = np.asarray(spikecounts)
    expected = (num_frames, num_trials)
    if spikecounts.shape != expected:
        raise ValueError(f"spikecounts has shape {spikecounts.shape}; expected {expected}")

    seed = running_seed
    st_sums = np.zeros((x * y, filter_length, num_trials), dtype=float)
    total_spikes = 0.0
    for trial in range(num_trials):
        if generate_stimuli:
            stimulus, seed = _make_stimulus(
                stimulus_generator, seed, x, y, num_frames
            )
            if save_stimuli:
                destination = _stimulus_file(save_stim_path, trial)
                destination.parent.mkdir(parents=True, exist_ok=True)
                with h5py.File(destination, "w") as handle:
                    handle.create_dataset("stim", data=stimulus, compression=3)
        else:
            stimulus = _load_stimulus(stim_path, trial)

        if stimulus.shape != (x * y, num_frames):
            raise ValueError(
                f"trial {trial} stimulus has shape {stimulus.shape}; "
                f"expected {(x * y, num_frames)}"
            )
        for frame in np.flatnonzero(spikecounts[:, trial]):
            if frame < filter_length:
                continue
            count = spikecounts[frame, trial]
            st_sums[:, :, trial] += count * stimulus[:, frame-filter_length:frame]
            total_spikes += count

    if total_spikes == 0:
        raise ValueError("no spikes occur after the requested filter length")
    sta = st_sums.sum(axis=2) / total_spikes
    sta_reshaped = sta.reshape(x, y, filter_length)
    max_x, max_y, max_t = np.unravel_index(
        np.abs(sta_reshaped).argmax(), sta_reshaped.shape
    )
    spatial_sta = sta_reshaped[:, :, max_t].copy()
    temp_sta = sta_reshaped[max_x, max_y, :].copy()
    spatial_sta /= np.linalg.norm(spatial_sta)
    temp_sta /= np.linalg.norm(temp_sta)

    if save_sta:
        destination = Path(save_sta_path)
        destination.mkdir(parents=True, exist_ok=True)
        names = {
            spatial_sta_filename or f"Cell {cell_num} Uncropped Spatial STA.h5":
                ("spatial_STA", spatial_sta),
            temp_sta_filename or f"Cell {cell_num} Uncropped Temp STA.h5":
                ("temp_STA", temp_sta),
            sta_filename or f"Cell {cell_num} Uncropped STA.h5": ("STA", sta),
        }
        for filename, (key, value) in names.items():
            with h5py.File(destination / filename, "w") as handle:
                handle.create_dataset(key, data=value, compression=3)
    return sta, spatial_sta, temp_sta


def crop_sta(spatial_sta, sigma_coeff=1.5):
    """Crop a spatial STA to a fitted Gaussian ellipse and return its mask."""
    ellipse = fitellipse(spatial_sta, sigma=sigma_coeff)
    x_min, x_max = round(min(ellipse[0])), round(max(ellipse[0]))
    y_min, y_max = round(min(ellipse[1])), round(max(ellipse[1]))
    # Keep slices in bounds; the notebook could wrap negative indices.
    x_min, y_min = max(0, x_min), max(0, y_min)
    x_max, y_max = min(spatial_sta.shape[0], x_max), min(spatial_sta.shape[1], y_max)
    if x_max <= x_min or y_max <= y_min:
        raise ValueError("the fitted ellipse produces an empty crop")
    crop = (slice(x_min, x_max), slice(y_min, y_max))
    cropped = np.asarray(spatial_sta)[crop]

    # Fill pixels between the two ellipse intersections in every column.
    ref = np.rint(ellipse.T - np.array([x_min, y_min])).astype(int)
    mask = np.zeros(cropped.shape, dtype=bool)
    for column in range(cropped.shape[1]):
        rows = ref[ref[:, 1] == column, 0]
        rows = rows[(rows >= 0) & (rows < cropped.shape[0])]
        if rows.size:
            mask[rows.min():rows.max() + 1, column] = True
        elif 0 < column < cropped.shape[1] - 1:
            mask[:, column] = mask[:, column - 1]
    if not mask.any():
        raise ValueError("the fitted ellipse contains no in-bounds pixels")
    return np.where(mask, cropped, 0), crop, mask


def gen_ses(
    x,
    y,
    spikecounts,
    temp_sta,
    sta_cropped,
    crop,
    ell_pixels,
    filter_length,
    num_frames,
    num_trials,
    stim_path=None,
    generate_stimuli=False,
    running_seed=None,
    stimulus_generator=None,
):
    """Create temporally filtered, ellipse-masked ensembles and binary labels."""
    if not generate_stimuli and stim_path is None:
        raise ValueError("stim_path is required when generate_stimuli=False")
    seed = running_seed
    examples, labels = [], []
    mask = np.asarray(ell_pixels, dtype=bool).ravel()
    crop_x, crop_y = np.asarray(sta_cropped).shape
    for trial in range(num_trials):
        if generate_stimuli:
            stimulus, seed = _make_stimulus(
                stimulus_generator, seed, x, y, num_frames
            )
        else:
            stimulus = _load_stimulus(stim_path, trial)
        stimulus = stimulus.reshape(x, y, num_frames)[crop].reshape(
            crop_x * crop_y, num_frames
        )
        for frame in range(filter_length, num_frames):
            frames = stimulus[:, frame-filter_length:frame]
            current = np.mean(frames * np.asarray(temp_sta), axis=1)
            examples.append(current[mask])
            labels.append(int(spikecounts[frame, trial] != 0))
    return np.asarray(examples), np.asarray(labels)


def gen_eses(
    x,
    y,
    spikecounts,
    temp_sta,
    sta_cropped,
    crop,
    ell_pixels,
    filter_length,
    num_frames,
    num_trials,
    stim_path,
):
    """Create ellipse-masked ensembles only at frames containing spikes.

    This matches the autoencoder notebook: a frame with one or more spikes
    contributes one example (spike multiplicity does not duplicate examples).
    """
    examples = []
    mask = np.asarray(ell_pixels, dtype=bool).ravel()
    crop_x, crop_y = np.asarray(sta_cropped).shape
    for trial in range(num_trials):
        stimulus = _load_stimulus(stim_path, trial)
        stimulus = stimulus.reshape(x, y, num_frames)[crop].reshape(
            crop_x * crop_y, num_frames
        )
        for frame in np.flatnonzero(spikecounts[:, trial]):
            if frame < filter_length:
                continue
            frames = stimulus[:, frame-filter_length:frame]
            current = np.mean(frames * np.asarray(temp_sta), axis=1)
            examples.append(current[mask])
    if not examples:
        raise ValueError("no spike-triggered ensembles were generated")
    return np.asarray(examples)


# Backwards-compatible names used by the exported notebook.
gen_STA = gen_sta
crop_STA = crop_sta
gen_SEs = gen_ses
gen_ESEs = gen_eses
