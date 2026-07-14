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
    """One-layer encoder/decoder architecture from the Colab notebook."""

    def __init__(self, pixel_num, node_num, output_activation="sigmoid"):
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
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    total, samples = 0.0, 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for ensembles in loader:
            ensembles = ensembles.to(device)
            if training:
                optimizer.zero_grad()
            reconstructed = model(ensembles)
            # Mean squared error is batch-size invariant, unlike the notebook's
            # unnormalised Frobenius norm.
            loss = nn.functional.mse_loss(reconstructed, ensembles)
            loss = loss + l1_coefficient * model.subunit_weights.abs().sum()
            if spatial_coefficient:
                loss = loss + spatial_coefficient * _spatial_group_penalty(
                    model.subunit_weights, crop_shape, ellipse_mask
                )
            if training:
                loss.backward()
                optimizer.step()
            total += loss.item() * len(ensembles)
            samples += len(ensembles)
    if samples == 0:
        raise ValueError("cannot train or evaluate on an empty dataset")
    return total / samples


def run_autoencoder(
    model, train_set, valid_set=None, *, batch_size=100, learning_rate=0.001,
    n_epochs=100, stop_threshold=1e-5, l1_coefficient=0.0,
    l2_coefficient=0.0, spatial_coefficient=0.0, crop_shape=None,
    ellipse_mask=None, scheduler_step=None, scheduler_gamma=0.95,
):
    """Train an autoencoder and restore its best validation-loss weights."""
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
    best_loss, best_state = float("inf"), None
    for epoch in range(n_epochs):
        train_loss = _epoch(
            model, train_loader, optimizer, l1_coefficient,
            spatial_coefficient, crop_shape, ellipse_mask,
        )
        losses["train"].append(train_loss)
        if valid_loader is not None:
            valid_loss = _epoch(
                model, valid_loader, None, l1_coefficient,
                spatial_coefficient, crop_shape, ellipse_mask,
            )
            losses["valid"].append(valid_loss)
            improvement = best_loss - valid_loss
            if valid_loss < best_loss:
                best_loss = valid_loss
                best_state = copy.deepcopy(model.state_dict())
            if epoch > 0 and improvement <= stop_threshold:
                break
        if scheduler is not None:
            scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses


# Notebook-compatible names.
Autoencoder = SubunitAutoencoder
MyDataset = EnsembleDataset
