"""PyTorch classifier used to recover candidate receptive-field subunits."""

import copy

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset


class SpikeDataset(Dataset):
    """Pair stimulus ensembles with binary spike/no-spike labels."""

    def __init__(self, x, y):
        self.x = torch.as_tensor(np.asarray(x), dtype=torch.float32)
        self.y = torch.as_tensor(np.asarray(y), dtype=torch.float32)
        if len(self.x) != len(self.y):
            raise ValueError("features and labels must contain the same number of samples")

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return len(self.x)


class SpikeClassifier(nn.Module):
    """One ReLU hidden layer followed by a binary logistic output."""

    def __init__(self, pixel_num, node_num):
        super().__init__()
        self.layer1 = nn.Linear(pixel_num, node_num)
        self.layer2 = nn.Linear(node_num, 1)

    def forward(self, inputs):
        return torch.sigmoid(self.layer2(torch.relu(self.layer1(inputs))))


def _l1_penalty(model):
    return model.layer1.weight.abs().sum()


def _run_epoch(model, data_loader, l1_coeff=0.0, optimizer=None):
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    total_loss = total_correct = total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for ensembles, labels in data_loader:
            ensembles, labels = ensembles.to(device), labels.to(device)
            if training:
                optimizer.zero_grad()
            probabilities = model(ensembles).flatten()
            loss = nn.functional.binary_cross_entropy(probabilities, labels.flatten())
            loss = loss + l1_coeff * _l1_penalty(model)
            if training:
                loss.backward()
                optimizer.step()
            batch_size = labels.numel()
            total_loss += loss.item() * batch_size
            total_correct += (probabilities.round() == labels.flatten()).sum().item()
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("cannot run a model on an empty dataset")
    return total_loss / total_samples, total_correct / total_samples


def run_model(
    model,
    running_mode="train",
    train_set=None,
    valid_set=None,
    test_set=None,
    sampler=None,
    batch_size=1,
    learning_rate=0.5,
    n_epochs=1,
    stop_thr=1e-5,
    L1_coeff=0,
    L2_coeff=0,
    shuffle=True,
    print_output=False,
    scheduler_step=None,
    scheduler_gamma=0.75,
):
    """Train or evaluate the classifier.

    When validation data are supplied, the best validation-loss weights are
    restored. Early stopping occurs only after the validation improvement is
    non-positive or smaller than ``stop_thr``; this fixes the notebook's loop,
    which compared training losses even when validation data were available.
    """
    if running_mode not in {"train", "test"}:
        raise ValueError("running_mode must be 'train' or 'test'")
    if sampler is not None:
        shuffle = False
    if running_mode == "test":
        if test_set is None:
            raise ValueError("test_set is required in test mode")
        loader = DataLoader(test_set, batch_size=batch_size, shuffle=shuffle)
        return _run_epoch(model, loader, L1_coeff)
    if train_set is None:
        raise ValueError("train_set is required in train mode")

    training = DataLoader(
        train_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler
    )
    validation = (
        DataLoader(valid_set, batch_size=batch_size, shuffle=False)
        if valid_set is not None else None
    )
    optimizer = torch.optim.SGD(
        model.parameters(), lr=learning_rate, weight_decay=L2_coeff
    )
    scheduler = (
        torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=scheduler_step, gamma=scheduler_gamma
        ) if scheduler_step else None
    )
    losses, accuracies = {"train": [], "valid": []}, {"train": [], "valid": []}
    best_loss, best_state = float("inf"), None
    for epoch in range(n_epochs):
        train_loss, train_accuracy = _run_epoch(
            model, training, L1_coeff, optimizer
        )
        losses["train"].append(train_loss)
        accuracies["train"].append(train_accuracy)
        if validation is not None:
            valid_loss, valid_accuracy = _run_epoch(model, validation, L1_coeff)
            losses["valid"].append(valid_loss)
            accuracies["valid"].append(valid_accuracy)
            improvement = best_loss - valid_loss
            if valid_loss < best_loss:
                best_loss, best_state = valid_loss, copy.deepcopy(model.state_dict())
            if epoch > 0 and improvement <= stop_thr:
                break
        if print_output:
            message = f"Epoch {epoch + 1}: loss={train_loss:.6g}, accuracy={train_accuracy:.3f}"
            if validation is not None:
                message += f", val_loss={valid_loss:.6g}, val_accuracy={valid_accuracy:.3f}"
            print(message)
        if scheduler is not None:
            scheduler.step()
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, accuracies


# Backwards-compatible class names used in earlier scripts.
Spike_Dataset = SpikeDataset
Spike_Classifier = SpikeClassifier


def _train(model, data_loader, optimizer, L1_coeff):
    loss, accuracy = _run_epoch(model, data_loader, L1_coeff, optimizer)
    return model, loss, accuracy


def _test(model, data_loader, L1_coeff):
    return _run_epoch(model, data_loader, L1_coeff)
