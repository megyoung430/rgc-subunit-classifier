"""End-to-end orchestration of STA construction and subunit classification."""

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from scipy import io as spio
from sklearn.model_selection import train_test_split
from torch.utils.data import WeightedRandomSampler

from ..analysis.analyze_subunits import get_subunits
from ..models.subunit_classifier import SpikeClassifier, SpikeDataset, run_model
from .generate_training_data import crop_sta, gen_ses, gen_sta


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


def _balanced_sampler(labels, generator):
    counts = np.bincount(np.asarray(labels, dtype=int), minlength=2)
    if np.any(counts == 0):
        raise ValueError("training data must contain both spike and no-spike examples")
    sample_weights = len(labels) / counts[np.asarray(labels, dtype=int)]
    return WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        len(sample_weights), replacement=True, generator=generator,
    )


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
    stop_threshold=0,
    l1_coefficient=0,
    l2_coefficient=0,
    random_state=0,
    max_trials=None,
    scheduler_step=None,
    scheduler_gamma=0.75,
    morans_threshold=0.25,
):
    """Run the reproducible training portion of the manuscript pipeline.

    Frozen-stimulus LN/subunit evaluation is intentionally exposed separately
    in :mod:`src.prediction`, because the repository does not include the
    required ``stim_frz.h5`` or per-trial stimulus files.
    """
    torch.manual_seed(random_state)
    rng = torch.Generator().manual_seed(random_state)
    spike_data = spio.loadmat(data_path)["spk1"]
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

    names = {
        "spatial_sta_filename": f"Cell {cell_num} Uncropped Spatial STA.h5",
        "temp_sta_filename": f"Cell {cell_num} Uncropped Temp STA.h5",
        "sta_filename": f"Cell {cell_num} Uncropped STA.h5",
    }
    sta, spatial_sta, temporal_sta = gen_sta(
        stimulus_width, stimulus_height, spikecounts, cell_num, filter_length,
        num_frames, num_trials, sta_path=sta_path, stim_path=stim_path, **names,
    )
    cropped_sta, crop, ellipse_mask = crop_sta(spatial_sta, sigma)
    examples, labels = gen_ses(
        stimulus_width, stimulus_height, spikecounts, temporal_sta, cropped_sta,
        crop, ellipse_mask, filter_length, num_frames, num_trials,
        stim_path=stim_path,
    )

    x_train, x_test, y_train, y_test = train_test_split(
        examples, labels, test_size=0.25, stratify=labels, random_state=random_state
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        x_train, y_train, test_size=0.25, stratify=y_train,
        random_state=random_state,
    )
    train, valid, test = (
        SpikeDataset(x_train, y_train), SpikeDataset(x_valid, y_valid),
        SpikeDataset(x_test, y_test),
    )
    model = SpikeClassifier(examples.shape[1], node_num)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    sampler = _balanced_sampler(y_train, rng)
    model, losses, accuracies = run_model(
        model, "train", train_set=train, valid_set=valid, sampler=sampler,
        batch_size=batch_size, learning_rate=learning_rate, n_epochs=num_epochs,
        stop_thr=stop_threshold, L1_coeff=l1_coefficient,
        L2_coeff=l2_coefficient,
        scheduler_step=scheduler_step, scheduler_gamma=scheduler_gamma,
    )
    test_loss, test_accuracy = run_model(
        model, "test", test_set=test, batch_size=batch_size,
        L1_coeff=l1_coefficient, shuffle=False,
    )

    compact_weights = model.layer1.weight.detach().cpu().numpy()
    node_weights = np.zeros((node_num, cropped_sta.size))
    node_weights[:, ellipse_mask.ravel()] = compact_weights
    subunits = get_subunits(
        node_weights, *cropped_sta.shape,
        threshold_morans_i=morans_threshold,
    )
    if subunits_save_path is not None:
        destination = Path(subunits_save_path)
        destination.mkdir(parents=True, exist_ok=True)
        suffix = "" if version == 0 else f" ({version})"
        filename = (
            f"Cell {cell_num}, {node_num} Nodes, {learning_rate} LR, "
            f"{l1_coefficient} L1 Coeff, {num_epochs} Epochs Subunits{suffix}.h5"
        )
        with h5py.File(destination / filename, "w") as handle:
            handle.create_dataset("subunits", data=subunits, compression=3)
    return PipelineResult(
        model, subunits, node_weights, sta, spatial_sta, temporal_sta, crop,
        ellipse_mask, losses, accuracies, test_loss, test_accuracy,
    )


def subunit_model(cell_num, ver, data_path, stim_path, **kwargs):
    """Compatibility wrapper for the notebook's original entry point."""
    aliases = {
        "STA_path": "sta_path", "learn_rate": "learning_rate",
        "stop_thr": "stop_threshold", "L1_reg": "l1_coefficient",
        "L2_reg": "l2_coefficient",
    }
    for old, new in aliases.items():
        if old in kwargs:
            kwargs[new] = kwargs.pop(old)
    result = run_subunit_model(
        cell_num, data_path, stim_path, version=ver, **kwargs
    )
    return result.model, result.subunits, result.crop, *result.ellipse_mask.shape
