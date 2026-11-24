"""PyTorch classifier used to recover candidate receptive-field subunits."""

import copy
from dataclasses import dataclass

import numpy as np
import torch
from sklearn.metrics import balanced_accuracy_score, roc_auc_score
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


@dataclass(frozen=True)
class ClassifierMetrics:
    """Held-out performance of a spike classifier.

    Spikes are rare (a few percent of frames), so plain accuracy is dominated
    by the no-spike frames; balanced accuracy (mean of the hit rates of the
    two classes, 0.5 = chance) and ROC AUC (0.5 = chance) are the honest
    measures.

    Attributes:
        loss: Mean binary cross-entropy at the true spike rate, without
            regularization penalties.
        balanced_loss: Binary cross-entropy averaged within each class and
            then across the two classes, i.e. as if spikes were as common as
            non-spikes, matching the balanced training (0.693 = chance).
        accuracy: Fraction of frames classified correctly at threshold 0.5.
        balanced_accuracy: Mean of the spike and no-spike hit rates.
        auc: Area under the ROC curve of the predicted spike probabilities
            (NaN if only one class is present).
    """

    loss: float
    balanced_loss: float
    accuracy: float
    balanced_accuracy: float
    auc: float


def _l1_penalty(model):
    return model.layer1.weight.abs().sum()


def _run_epoch(model, data_loader, l1_coeff=0.0, optimizer=None):
    """One pass; trains when ``optimizer`` is given.

    Returns the mean data loss (binary cross-entropy, without the L1 term,
    which is only added to the training objective) and the accuracy.
    """
    training = optimizer is not None
    model.train(training)
    device = next(model.parameters()).device
    total_loss = total_correct = total_samples = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for ensembles, labels in data_loader:
            ensembles, labels = ensembles.to(device), labels.to(device).flatten()
            probabilities = model(ensembles).flatten()
            data_loss = nn.functional.binary_cross_entropy(probabilities, labels)
            if training:
                optimizer.zero_grad()
                (data_loss + l1_coeff * _l1_penalty(model)).backward()
                optimizer.step()
            batch_size = labels.numel()
            total_loss += data_loss.item() * batch_size
            total_correct += (probabilities.round() == labels).sum().item()
            total_samples += batch_size
    if total_samples == 0:
        raise ValueError("cannot run a model on an empty dataset")
    return total_loss / total_samples, total_correct / total_samples


def _balanced_bce(probabilities, labels):
    """Binary cross-entropy averaged within each class, then across the classes."""
    losses = nn.functional.binary_cross_entropy(probabilities, labels, reduction="none")
    present = [losses[labels == value].mean() for value in (0.0, 1.0) if (labels == value).any()]
    return torch.stack(present).mean()


def _predict(model, dataset, batch_size):
    """Spike probabilities and labels for every example of ``dataset``."""
    model.eval()
    device = next(model.parameters()).device
    probabilities, labels = [], []
    with torch.no_grad():
        for ensembles, batch_labels in DataLoader(dataset, batch_size=batch_size):
            probabilities.append(model(ensembles.to(device)).flatten().cpu())
            labels.append(batch_labels.flatten())
    if not labels:
        raise ValueError("cannot evaluate a model on an empty dataset")
    return torch.cat(probabilities), torch.cat(labels)


def evaluate_classifier(model, dataset, batch_size=1024):
    """Losses, accuracy, balanced accuracy and ROC AUC of ``model`` on ``dataset``."""
    p, y = _predict(model, dataset, batch_size)
    loss = nn.functional.binary_cross_entropy(p, y).item()
    balanced_loss = _balanced_bce(p, y).item()
    p, y = p.numpy(), y.numpy().astype(int)
    predicted = (p >= 0.5).astype(int)
    auc = roc_auc_score(y, p) if len(np.unique(y)) == 2 else float("nan")
    return ClassifierMetrics(
        loss=loss,
        balanced_loss=balanced_loss,
        accuracy=float((predicted == y).mean()),
        balanced_accuracy=float(balanced_accuracy_score(y, predicted)),
        auc=float(auc),
    )


def run_model(
    model,
    train_set,
    valid_set=None,
    sampler=None,
    batch_size=1,
    learning_rate=0.5,
    n_epochs=1,
    patience=None,
    min_improvement=0.0,
    L1_coeff=0,
    L2_coeff=0,
    shuffle=True,
    print_output=False,
    scheduler_step=None,
    scheduler_gamma=0.75,
):
    """Train the classifier with SGD.

    The validation loss is the class-balanced binary cross-entropy (averaged
    within spike and no-spike frames, then across the two), matching the
    balanced training: at the true spike rate of a few percent, plain
    cross-entropy mostly rewards predicting "no spike" and swings widely from
    epoch to epoch. With validation data, the weights with the lowest
    validation loss are restored at the end. Every epoch is run unless
    ``patience`` is given, in which case training stops once the validation
    loss has failed to improve by more than ``min_improvement`` for
    ``patience`` consecutive epochs. Reported losses exclude the L1 term; L1
    and L2 only shape the training objective. Training losses are at the
    sampled (balanced) rate when a balanced sampler is used.

    Raises:
        ValueError: If ``patience`` is given without ``valid_set``.

    Returns:
        (model, losses, accuracies), the last two as ``{"train": [...],
        "valid": [...]}`` per epoch.
    """
    if train_set is None:
        raise ValueError("train_set is required")
    if patience is not None and valid_set is None:
        raise ValueError("patience needs a valid_set to measure improvement on")
    if n_epochs < 1:
        raise ValueError(f"n_epochs must be at least 1, not {n_epochs}")
    if sampler is not None:
        shuffle = False
    training = DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, weight_decay=L2_coeff)
    scheduler = (
        torch.optim.lr_scheduler.StepLR(optimizer, step_size=scheduler_step, gamma=scheduler_gamma)
        if scheduler_step else None
    )
    losses, accuracies = {"train": [], "valid": []}, {"train": [], "valid": []}
    best_loss, best_state, epochs_without_improvement = float("inf"), None, 0
    for epoch in range(n_epochs):
        train_loss, train_accuracy = _run_epoch(model, training, L1_coeff, optimizer)
        losses["train"].append(train_loss)
        accuracies["train"].append(train_accuracy)
        message = f"Epoch {epoch + 1}: loss={train_loss:.6g}, accuracy={train_accuracy:.3f}"
        if valid_set is not None:
            p, y = _predict(model, valid_set, max(batch_size, 1024))
            valid_loss = _balanced_bce(p, y).item()
            valid_accuracy = float(((p >= 0.5).float() == y).float().mean())
            losses["valid"].append(valid_loss)
            accuracies["valid"].append(valid_accuracy)
            message += f", val_loss={valid_loss:.6g}, val_accuracy={valid_accuracy:.3f}"
            if valid_loss < best_loss - min_improvement:
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            if valid_loss < best_loss:
                best_loss, best_state = valid_loss, copy.deepcopy(model.state_dict())
        if print_output:
            print(message)
        if scheduler is not None:
            scheduler.step()
        if patience is not None and epochs_without_improvement >= patience:
            break
    if best_state is not None:
        model.load_state_dict(best_state)
    return model, losses, accuracies


# Notebook-compatible class names.
Spike_Dataset = SpikeDataset
Spike_Classifier = SpikeClassifier
