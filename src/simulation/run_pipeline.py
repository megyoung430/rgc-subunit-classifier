"""End-to-end orchestration of STA construction and subunit classification."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import WeightedRandomSampler

from ..analysis.analyze_subunits import get_subunits
from ..data_cache import cropped_stimulus, load_spike_data
from ..models.subunit_classifier import (
    SpikeClassifier,
    SpikeDataset,
    evaluate_classifier,
    run_model,
)
from ..outputs import write_new_file
from .generate_training_data import crop_sta, gen_ses, gen_sta
from .splits import split_by_trial


@dataclass
class PipelineResult:
    """The fitted model and the intermediate values needed for interpretation."""

    model: SpikeClassifier
    subunits: np.ndarray
    node_weights: np.ndarray
    sta: np.ndarray
    spatial_sta: np.ndarray
    temporal_sta: np.ndarray
    crop: tuple[slice, slice]
    ellipse_mask: np.ndarray
    losses: dict
    accuracies: dict
    test_loss: float
    test_accuracy: float
    test_balanced_accuracy: float
    test_auc: float
    split_trials: dict


def _subfolder(cache_dir, name):
    return None if cache_dir is None else Path(cache_dir) / name


def _balanced_sampler(labels, generator):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if np.any(counts == 0):
        raise ValueError("training data must contain both spike and no-spike examples")
    sample_weights = len(labels) / counts[np.asarray(labels, dtype=int)]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        len(sample_weights), replacement=True, generator=generator,
    )


def save_subunits(path, subunits):
    """Write subunits to a new HDF5 file; never overwrites an existing file."""

    def write(temporary):
        with h5py.File(temporary, "w") as handle:
            handle.create_dataset("subunits", data=subunits, compression=3)

    return write_new_file(path, write)


def run_subunit_model(
    cell_num,
    data_path,
    stim_path,
    *,
    sta_path=None,
    subunits_save_path=None,
    version=0,
    stimulus_width=200,
    stimulus_height=150,
    refresh_rate=30,
    filter_length_seconds=0.67,
    sigma=3,
    node_num=60,
    batch_size=25,
    learning_rate=0.5,
    num_epochs=100,
    patience=None,
    min_improvement=0.0,
    l1_coefficient=0,
    l2_coefficient=0,
    random_state=0,
    max_trials=None,
    cache_dir=None,
    scheduler_step=None,
    scheduler_gamma=0.75,
    morans_threshold=0.25,
):
    """Train the spike classifier on one cell and select its subunits.

    Examples are split into training, validation and test sets by whole
    trials (:func:`src.simulation.splits.split_by_trial`), so no test frame
    shares stimulus history with a training frame. Training draws spike and
    no-spike frames equally often; the test set keeps the true spike rate,
    so judge it by ``test_balanced_accuracy`` or ``test_auc`` rather than
    ``test_accuracy``. Every epoch is run unless ``patience`` is given.

    Frozen-stimulus LN/subunit evaluation is exposed separately in
    :mod:`src.prediction`. ``sta_path`` caches the STA; ``cache_dir`` caches
    the spike counts (``spikes/``) and this cell's cropped stimulus
    (``stimuli/``), so later runs of the same cell skip reading the trial files.
    """
    torch.manual_seed(random_state)
    rng = torch.Generator().manual_seed(random_state)
    spike_data = load_spike_data(
        data_path, _subfolder(cache_dir, "spikes"), keys=("spk1",)
    )["spk1"]
    if not 0 <= cell_num < spike_data.shape[0]:
        raise IndexError(f"cell_num must be between 0 and {spike_data.shape[0] - 1}")
    spikecounts = spike_data[cell_num]
    num_frames, num_trials = spikecounts.shape
    available = [Path(stim_path) / f"stim{i:04d}.h5" for i in range(num_trials)]
    if max_trials is None:
        missing = [path.name for path in available if not path.exists()]
        if missing:
            preview = ", ".join(missing[:5])
            raise FileNotFoundError(
                f"{len(missing)} of {num_trials} stimulus trials are missing "
                f"from {stim_path} (first: {preview}); pass max_trials only for "
                "an explicitly partial analysis"
            )
    else:
        num_trials = min(int(max_trials), num_trials)
        spikecounts = spikecounts[:, :num_trials]
        missing = [path.name for path in available[:num_trials] if not path.exists()]
        if missing:
            raise FileNotFoundError(f"missing requested stimulus file {missing[0]}")
    filter_length = int(filter_length_seconds * refresh_rate)

    sta, spatial_sta, temporal_sta = gen_sta(
        stimulus_width, stimulus_height, spikecounts, cell_num, filter_length,
        num_frames, num_trials, sta_path=sta_path, stim_path=stim_path,
    )
    cropped_sta, crop, ellipse_mask = crop_sta(spatial_sta, sigma)
    stimuli = cropped_stimulus(
        stim_path, crop, stimulus_width, stimulus_height, num_frames, num_trials,
        _subfolder(cache_dir, "stimuli"),
    )
    examples, labels = gen_ses(
        stimulus_width, stimulus_height, spikecounts, temporal_sta, cropped_sta,
        crop, ellipse_mask, filter_length, num_frames, num_trials,
        stimuli=stimuli,
    )

    trial_ids = np.repeat(np.arange(num_trials), num_frames - filter_length)
    train_index, valid_index, test_index = split_by_trial(trial_ids, random_state)
    train, valid, test = (
        SpikeDataset(examples[i], labels[i])
        for i in (train_index, valid_index, test_index)
    )
    y_train = labels[train_index]
    model = SpikeClassifier(examples.shape[1], node_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sampler = _balanced_sampler(y_train, rng)
    model, losses, accuracies = run_model(
        model, train_set=train, valid_set=valid, sampler=sampler,
        batch_size=batch_size, learning_rate=learning_rate, n_epochs=num_epochs,
        patience=patience, min_improvement=min_improvement,
        L1_coeff=l1_coefficient, L2_coeff=l2_coefficient,
        scheduler_step=scheduler_step, scheduler_gamma=scheduler_gamma,
    )
    metrics = evaluate_classifier(model, test)

    compact_weights = model.layer1.weight.detach().cpu().numpy()
    node_weights = np.zeros((node_num, cropped_sta.size))
    node_weights[:, ellipse_mask.ravel()] = compact_weights
    subunits = get_subunits(
        node_weights, *cropped_sta.shape,
        threshold_morans_i=morans_threshold,
    )
    if subunits_save_path is not None:
        suffix = "" if version == 0 else f" ({version})"
        filename = (
            f"Cell {cell_num}, {node_num} Nodes, {learning_rate} LR, "
            f"{l1_coefficient} L1 Coeff, {num_epochs} Epochs Subunits{suffix}.h5"
        )
        save_subunits(Path(subunits_save_path) / filename, subunits)
    return PipelineResult(
        model, subunits, node_weights, sta, spatial_sta, temporal_sta, crop,
        ellipse_mask, losses, accuracies, metrics.loss, metrics.accuracy,
        metrics.balanced_accuracy, metrics.auc,
        {name: np.unique(trial_ids[index]) for name, index in
         (("train", train_index), ("valid", valid_index), ("test", test_index))},
    )


def subunit_model(cell_num, ver, data_path, stim_path, **kwargs):
    """Compatibility wrapper for the notebook's original entry point."""
    aliases = {
        "STA_path": "sta_path", "learn_rate": "learning_rate",
        "L1_reg": "l1_coefficient",
        "L2_reg": "l2_coefficient",
    }
    for old, new in aliases.items():
        if old in kwargs:
            kwargs[new] = kwargs.pop(old)
    result = run_subunit_model(
        cell_num, data_path, stim_path, version=ver, **kwargs
    )
    return result.model, result.subunits, result.crop, *result.ellipse_mask.shape
