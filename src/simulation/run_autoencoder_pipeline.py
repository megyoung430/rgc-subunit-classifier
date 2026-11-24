"""End-to-end autoencoder subunit-extraction pipeline."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from ..analysis.analyze_subunits import get_subunits
from ..data_cache import cropped_stimulus, load_spike_data
from ..models.subunit_autoencoder import (
    EnsembleDataset,
    SubunitAutoencoder,
    evaluate_autoencoder,
    run_autoencoder,
)
from .generate_training_data import crop_sta, gen_eses, gen_sta
from .run_pipeline import _subfolder
from .splits import split_by_trial


@dataclass
class AutoencoderResult:
    """Outputs aligned with the classifier pipeline for subunit comparison."""

    model: SubunitAutoencoder
    subunits: np.ndarray
    node_weights: np.ndarray
    sta: np.ndarray
    spatial_sta: np.ndarray
    temporal_sta: np.ndarray
    crop: tuple[slice, slice]
    ellipse_mask: np.ndarray
    losses: dict
    validation_loss: float
    test_loss: float
    n_examples: int
    split_trials: dict


def run_autoencoder_model(
    cell_num,
    data_path,
    stim_path,
    *,
    sta_path=None,
    stimulus_width=200,
    stimulus_height=150,
    refresh_rate=30,
    filter_length_seconds=0.67,
    sigma=3,
    node_num=60,
    batch_size=100,
    learning_rate=0.001,
    num_epochs=100,
    patience=None,
    min_improvement=0.0,
    l1_coefficient=0,
    l2_coefficient=0,
    spatial_coefficient=0,
    output_activation="linear",
    random_state=0,
    max_trials=None,
    morans_threshold=0.25,
    cache_dir=None,
):
    """Extract candidate subunits using spike-triggered reconstruction.

    The returned ``node_weights`` and ``subunits`` have exactly the same shapes
    and Moran's-I selection semantics as ``run_subunit_model``; ``sta_path``
    and ``cache_dir`` cache inputs in the same way. Spike-triggered ensembles
    are split into training, validation and test sets by whole trials.
    ``validation_loss`` is the lowest validation reconstruction error (the
    restored weights'); ``test_loss`` is the error on the held-out test trials.
    Every epoch is run unless ``patience`` is given.
    """
    torch.manual_seed(random_state)
    spike_data = load_spike_data(
        data_path, _subfolder(cache_dir, "spikes"), keys=("spk1",)
    )["spk1"]
    if not 0 <= cell_num < len(spike_data):
        raise IndexError(f"cell_num must be between 0 and {len(spike_data) - 1}")
    spikecounts = spike_data[cell_num]
    num_frames, total_trials = spikecounts.shape
    num_trials = total_trials if max_trials is None else min(max_trials, total_trials)
    paths = [Path(stim_path) / f"stim{i:04d}.h5" for i in range(num_trials)]
    missing = [path.name for path in paths if not path.exists()]
    if missing:
        message = f"{len(missing)} requested stimulus trials are missing (first: {missing[0]})"
        if max_trials is None:
            message += "; pass max_trials only for an explicitly partial analysis"
        raise FileNotFoundError(message)
    spikecounts = spikecounts[:, :num_trials]
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
    ensembles, trial_ids = gen_eses(
        stimulus_width, stimulus_height, spikecounts, temporal_sta, cropped_sta,
        crop, ellipse_mask, filter_length, num_frames, num_trials,
        stimuli=stimuli,
    )
    train_index, valid_index, test_index = split_by_trial(trial_ids, random_state)
    train, valid, test = (
        EnsembleDataset(ensembles[i]) for i in (train_index, valid_index, test_index)
    )
    model = SubunitAutoencoder(
        ensembles.shape[1], node_num, output_activation=output_activation
    ).to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    model, losses = run_autoencoder(
        model, train, valid, batch_size=batch_size,
        learning_rate=learning_rate, n_epochs=num_epochs,
        patience=patience, min_improvement=min_improvement,
        l1_coefficient=l1_coefficient,
        l2_coefficient=l2_coefficient, spatial_coefficient=spatial_coefficient,
        crop_shape=cropped_sta.shape, ellipse_mask=ellipse_mask,
    )
    compact = model.subunit_weights.detach().cpu().numpy()
    node_weights = np.zeros((node_num, cropped_sta.size))
    node_weights[:, ellipse_mask.ravel()] = compact
    subunits = get_subunits(
        node_weights, *cropped_sta.shape,
        threshold_morans_i=morans_threshold,
    )
    return AutoencoderResult(
        model, subunits, node_weights, sta, spatial_sta, temporal_sta, crop,
        ellipse_mask, losses, min(losses["valid"]), evaluate_autoencoder(model, test),
        len(ensembles),
        {name: np.unique(trial_ids[index]) for name, index in
         (("train", train_index), ("valid", valid_index), ("test", test_index))},
    )
