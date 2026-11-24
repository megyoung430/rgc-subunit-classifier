"""Load spike counts and cropped stimuli once, with optional on-disk caches.

Only a cell's ellipse bounding box of the 200 × 150 stimulus is ever used, so
the training-data and prediction code take each cell's stimulus as one array
cropped to that box, read once. Given a ``cache_dir``, the loaders also keep
what they read on disk, so later runs of the same cell (other seeds, sweeps,
the prediction fits) skip reading the trial files. Cache keys include the
stimulus folder and each file's name, size and modification time, so another
stimulus folder is never served a cached array. Cache files are written whole
or not at all and never overwritten; delete the folder to rebuild.
"""

import functools
import hashlib
import json
import re
from pathlib import Path

import h5py
import numpy as np
from scipy import io as spio

from .outputs import check_output_dir, write_new_file

# Bump when a cached layout changes, so older cache files are not reused.
CACHE_VERSION = 2
SPIKE_KEYS = ("spk1", "spk2")
TRIAL_FILE = re.compile(r"stim\d{4}\.h5")


def _write_cache(path, save):
    """Write a cache file; if another run wrote it meanwhile, keep theirs."""
    try:
        write_new_file(path, save)
    except FileExistsError:
        pass


def _file_digest(path):
    digest = hashlib.sha256(f"v{CACHE_VERSION}".encode())
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()[:16]


@functools.lru_cache(maxsize=4)
def _stored_spikes(resolved, digest, cache_dir):
    """Every spike variable the file holds, as read-only arrays (memoized by content)."""

    def from_mat():
        present = {name for name, _, _ in spio.whosmat(resolved)}
        names = [key for key in SPIKE_KEYS if key in present]
        loaded = spio.loadmat(resolved, variable_names=names) if names else {}
        return {key: loaded[key] for key in names}

    if cache_dir is None:
        arrays = from_mat()
    else:
        cache = Path(cache_dir) / f"spikes {digest}.npz"
        if not cache.exists():
            arrays = from_mat()
            _write_cache(cache, lambda temporary: np.savez_compressed(temporary, **arrays))
        with np.load(cache) as stored:
            arrays = {key: stored[key] for key in stored.files}
    for array in arrays.values():
        array.setflags(write=False)
    return arrays


def load_spike_data(data_path, cache_dir=None, keys=SPIKE_KEYS):
    """Return ``{key: spike counts}`` for each requested key of the cell ``.mat`` file.

    The arrays are read-only and shared between calls with the same file
    content; the returned dict is new on every call. With ``cache_dir``, the
    spike variables are also kept as a compressed ``.npz`` named by a hash of
    the ``.mat`` file's content, so an edited file is never served from an
    old cache.

    Raises:
        ValueError: If the file has no variable named by one of ``keys``.
    """
    path = Path(data_path).resolve()
    cache = None if cache_dir is None else str(check_output_dir(cache_dir).resolve())
    stored = _stored_spikes(str(path), _file_digest(path), cache)
    missing = [key for key in keys if key not in stored]
    if missing:
        raise ValueError(f"{data_path} has no variable {', '.join(missing)}")
    return {key: stored[key] for key in keys}


def _stat_entry(path):
    stat = path.stat()
    return [path.name, stat.st_size, stat.st_mtime_ns]


def _trial_files_fingerprint(stim_path):
    folder = Path(stim_path).resolve()
    files = sorted(p for p in folder.iterdir() if TRIAL_FILE.fullmatch(p.name))
    return {"folder": str(folder), "files": [_stat_entry(p) for p in files]}


def _key(settings):
    text = json.dumps({"v": CACHE_VERSION, **settings}, sort_keys=True)
    return hashlib.sha256(text.encode()).hexdigest()[:16]


def _bounds(crop):
    return [[part.start, part.stop] for part in crop]


def _crop_label(crop):
    (x0, x1), (y0, y1) = _bounds(crop)
    return f"x{x0}-{x1} y{y0}-{y1}"


def check_stimuli(stimuli, n_pixels, num_frames, num_trials):
    """Raise if a preloaded cropped stimulus does not fit the requested crop and trials.

    ``stimuli`` must have shape ``(trials, crop pixels, frames)`` with at least
    ``num_trials`` trials.

    Raises:
        ValueError: On a mismatch in pixels, frames or trials.
    """
    shape = np.shape(stimuli)
    if len(shape) != 3 or shape[1:] != (n_pixels, num_frames):
        raise ValueError(
            f"stimuli has shape {shape}; expected (trials, {n_pixels} pixels, "
            f"{num_frames} frames)"
        )
    if shape[0] < num_trials:
        raise ValueError(f"stimuli has {shape[0]} trials; {num_trials} trials are needed")


def _read_cropped_trials(stim_path, crop, x, y, num_frames, num_trials):
    n_pixels = (crop[0].stop - crop[0].start) * (crop[1].stop - crop[1].start)
    cropped = None
    for trial in range(num_trials):
        path = Path(stim_path) / f"stim{trial:04d}.h5"
        if not path.exists():
            raise FileNotFoundError(f"stimulus file {path} is missing")
        with h5py.File(path, "r") as handle:
            stimulus = handle["stim"][:]
        if cropped is None:
            cropped = np.empty((num_trials, n_pixels, num_frames), dtype=stimulus.dtype)
        elif stimulus.dtype != cropped.dtype:
            raise ValueError(f"{path} has dtype {stimulus.dtype}; earlier trials {cropped.dtype}")
        cropped[trial] = stimulus.reshape(x, y, num_frames)[crop].reshape(n_pixels, num_frames)
    return cropped


def _cached_trial_counts(cache_dir, prefix):
    pattern = re.compile(re.escape(prefix) + r" (\d+) trials\.npy")
    for path in Path(cache_dir).glob(f"{prefix} * trials.npy"):
        match = pattern.fullmatch(path.name)
        if match:
            yield int(match.group(1)), path


def cropped_stimulus(stim_path, crop, x, y, num_frames, num_trials, cache_dir=None):
    """Return the running-noise stimulus cropped to ``crop`` for the first ``num_trials``.

    The array has shape ``(num_trials, crop pixels, num_frames)``, trial first so
    each trial is one contiguous block, with pixels in the order the
    training-data and prediction code use, and the stimulus files' own dtype.
    With ``cache_dir`` it is stored as an ``.npy`` (memory-mapped on reuse)
    keyed on the crop, stimulus size and the stimulus folder's trial files;
    a request for fewer trials is served from a cached file with more.
    """
    if cache_dir is None:
        return _read_cropped_trials(stim_path, crop, x, y, num_frames, num_trials)
    key = _key({
        "crop": _bounds(crop), "x": x, "y": y, "frames": num_frames,
        "stimulus": _trial_files_fingerprint(stim_path),
    })
    prefix = f"stimulus {key} {_crop_label(crop)}"
    folder = check_output_dir(cache_dir)
    usable = [(n, p) for n, p in _cached_trial_counts(folder, prefix) if n >= num_trials]
    if usable:
        _, path = min(usable)
        return np.load(path, mmap_mode="r")[:num_trials]
    cropped = _read_cropped_trials(stim_path, crop, x, y, num_frames, num_trials)
    path = folder / f"{prefix} {num_trials} trials.npy"
    _write_cache(path, lambda temporary: np.save(temporary, cropped))
    return np.load(path, mmap_mode="r")


def cropped_frozen_stimulus(stim_frz_path, crop, cache_dir=None):
    """Return the frozen-noise stimulus cropped to ``crop``, shape ``(pixels, frames)``."""
    source = Path(stim_frz_path).resolve() / "stim_frz.h5"

    def read():
        with h5py.File(source, "r") as handle:
            frozen = handle["stim_frz"][:]
        cropped = frozen[crop]
        return cropped.reshape(-1, cropped.shape[-1])

    if cache_dir is None:
        return read()
    key = _key({"crop": _bounds(crop), "frozen": [str(source), *_stat_entry(source)]})
    path = check_output_dir(cache_dir) / f"frozen stimulus {key} {_crop_label(crop)}.npy"
    if not path.exists():
        values = read()
        _write_cache(path, lambda temporary: np.save(temporary, values))
    return np.load(path, mmap_mode="r")
