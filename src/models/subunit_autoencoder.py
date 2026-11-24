"""Autoencoder alternative for extracting receptive-field subunits."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class EnsembleDataset(Dataset):
    """Unlabelled spike-triggered ensembles for reconstruction training."""

    def __init__(self, ensembles):
        self.ensembles = torch.as_tensor(
            np.asarray(ensembles), dtype=torch.float32
        )

    def __getitem__(self, index):
        return self.ensembles[index]

    def __len__(self):
        return len(self.ensembles)


class SubunitAutoencoder(nn.Module):
    """One-layer encoder/decoder; the encoder weights are candidate subunits.

    The output is linear by default because the ensembles are zero-centred:
    a sigmoid output (0 to 1) cannot reconstruct their negative values.
    """

    def __init__(self, pixel_num, node_num, output_activation="linear"):
        super().__init__()
        if output_activation not in {"sigmoid", "linear"}:
            raise ValueError("output_activation must be 'sigmoid' or 'linear'")
        self.encoder_layer = nn.Linear(pixel_num, node_num)
        self.decoder_layer = nn.Linear(node_num, pixel_num)
        self.output_activation = output_activation

    def encode(self, inputs):
        return torch.relu(self.encoder_layer(inputs))

    def forward(self, inputs):
        decoded = self.decoder_layer(self.encode(inputs))
        return torch.sigmoid(decoded) if self.output_activation == "sigmoid" else decoded

    @property
    def subunit_weights(self):
        """Encoder weights in the same node × pixel form as the classifier."""
        return self.encoder_layer.weight


def _spatial_group_penalty(weights, crop_shape, ellipse_mask, epsilon=0.01):
    """Differentiable version of the notebook's neighbour-aware penalty."""
    mask = torch.as_tensor(ellipse_mask, device=weights.device, dtype=torch.bool)
    full = weights.new_zeros((len(weights), *crop_shape))
    full[:, mask] = weights
    absolute = full.abs()
    padded = nn.functional.pad(absolute[:, None], (1, 1, 1, 1))
    neighbours = torch.zeros_like(absolute)
    for dx in range(3):
        for dy in range(3):
            if (dx, dy) != (1, 1):
                neighbours += padded[:, 0, dx:dx+crop_shape[0], dy:dy+crop_shape[1]]
    return (absolute[:, mask] / (epsilon + neighbours[:, mask])).sum()


def _epoch(model, loader, optimizer=None, l1_coefficient=0.0,
           spatial_coefficient=0.0, crop_shape=None, ellipse_mask=None):
    """One pass; trains when ``optimizer`` is given.

    Returns the mean reconstruction error (MSE). The L1 and spatial penalties
    are added to the training objective only, so reported losses can be
    compared across regularization settings.
    """
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    total, samples = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for ensembles in loader:
            ensembles = ensembles.to(device)
            # Mean squared error is batch-size invariant, unlike a summed norm.
            data_loss = nn.functional.mse_loss(model(ensembles), ensembles)
            if training:
                objective = data_loss + l1_coefficient * model.subunit_weights.abs().sum()
                if spatial_coefficient:
                    objective = objective + spatial_coefficient * _spatial_group_penalty(
                        model.subunit_weights, crop_shape, ellipse_mask
                    )
                optimizer.zero_grad()
                objective.backward()
                optimizer.step()
            total += data_loss.item() * len(ensembles)
            samples += len(ensembles)
    if samples == 0:
        raise ValueError("cannot train or evaluate on an empty dataset")
    return total / samples


def evaluate_autoencoder(model, dataset, batch_size=1024):
    """Mean reconstruction error (MSE) of ``model`` on ``dataset``."""
    return _epoch(model, DataLoader(dataset, batch_size=batch_size, shuffle=False))


def run_autoencoder(
    model, train_set, valid_set=None, *, batch_size=100, learning_rate=0.001,
    n_epochs=100, patience=None, min_improvement=0.0, l1_coefficient=0.0,
    l2_coefficient=0.0, spatial_coefficient=0.0, crop_shape=None,
    ellipse_mask=None, scheduler_step=None, scheduler_gamma=0.95,
):
    """Train an autoencoder with Adam.

    With validation data, the weights with the lowest validation loss are
    restored at the end. Every epoch is run unless ``patience`` is given, in
    which case training stops once the validation loss has failed to improve
    by more than ``min_improvement`` for ``patience`` consecutive epochs.
    Reported losses are reconstruction errors without the penalties.

    Returns:
        (model, losses) with losses as ``{"train": [...], "valid": [...]}``.
    """
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = (
        DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        if valid_set is not None else None
    )
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=l2_coefficient
    )
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        ) if scheduler_step else None
    )
    losses = {"train": [], "valid": []}
    best_loss, best_state, epochs_without_improvement = float("inf"), None, 0
    for _ in range(n_epochs):
        losses["train"].append(_epoch(
            model, train_loader, optimizer, l1_coefficient,
            spatial_coefficient, crop_shape, ellipse_mask,
        ))
        if valid_loader is not None:
            valid_loss = _epoch(model, valid_loader)
            losses["valid"].append(valid_loss)
            if valid_loss < best_loss - min_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if valid_loss < best_loss:
                best_loss, best_state = valid_loss, copy.deepcopy(model.state_dict())
        if scheduler is not None:
            scheduler.step()
        if patience is not None and epochs_without_improvement >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses


# Notebook-compatible names.
Autoencoder = SubunitAutoencoder
MyDataset = EnsembleDataset
