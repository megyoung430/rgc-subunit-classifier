import numpy as np
import pytest
import torch
from torch import nn

from src.models.subunit_classifier import (
    SpikeClassifier,
    SpikeDataset,
    evaluate_classifier,
    run_model,
)


def separable(n=200, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 2))
    return x, (x[:, 0] + 0.2 * rng.normal(size=n) > 0).astype(int)


def test_classifier_learns_and_reports_balanced_metrics():
    torch.manual_seed(0)
    x, y = separable()
    data = SpikeDataset(x, y)
    model, losses, accuracies = run_model(
        SpikeClassifier(2, 3), train_set=data, valid_set=data, batch_size=20,
        learning_rate=0.5, n_epochs=20,
    )
    metrics = evaluate_classifier(model, data)
    assert len(losses["train"]) == 20
    assert metrics.accuracy > 0.85
    assert metrics.balanced_accuracy > 0.85
    assert metrics.auc > 0.9


def test_balanced_accuracy_exposes_a_majority_class_guesser():
    model = SpikeClassifier(1, 1)
    with torch.no_grad():  # always predicts "no spike"
        model.layer1.weight.zero_(), model.layer1.bias.zero_()
        model.layer2.weight.zero_(), model.layer2.bias.fill_(-5.0)
    y = np.array([0] * 95 + [1] * 5)
    metrics = evaluate_classifier(model, SpikeDataset(np.zeros((100, 1)), y))
    assert metrics.accuracy == pytest.approx(0.95)
    assert metrics.balanced_accuracy == pytest.approx(0.5)


def test_all_epochs_run_unless_patience_is_given():
    torch.manual_seed(0)
    x, y = separable()
    data = SpikeDataset(x, y)
    _, losses, _ = run_model(SpikeClassifier(2, 3), train_set=data, valid_set=data,
                             batch_size=20, learning_rate=5.0, n_epochs=15)
    assert len(losses["valid"]) == 15
    _, losses, _ = run_model(SpikeClassifier(2, 3), train_set=data, valid_set=data,
                             batch_size=20, learning_rate=0.0, n_epochs=15, patience=3)
    assert len(losses["valid"]) == 4  # first epoch sets the best, then 3 without improvement


def test_reported_losses_exclude_the_l1_penalty():
    torch.manual_seed(0)
    x, y = separable(50)
    data = SpikeDataset(x, y)
    model, losses, _ = run_model(SpikeClassifier(2, 3), train_set=data, valid_set=data,
                                 batch_size=50, learning_rate=0.0, n_epochs=1, L1_coeff=10.0)
    with torch.no_grad():
        p = model(torch.as_tensor(x, dtype=torch.float32)).flatten()
        bce = nn.functional.binary_cross_entropy(p, torch.as_tensor(y, dtype=torch.float32))
    assert losses["valid"][0] == pytest.approx(bce.item(), rel=1e-5)
    assert evaluate_classifier(model, data).loss == pytest.approx(bce.item(), rel=1e-5)


def test_best_validation_weights_are_restored():
    torch.manual_seed(0)
    x, y = separable()
    data = SpikeDataset(x, y)
    model, losses, _ = run_model(SpikeClassifier(2, 3), train_set=data, valid_set=data,
                                 batch_size=20, learning_rate=5.0, n_epochs=15)
    assert evaluate_classifier(model, data).loss == pytest.approx(min(losses["valid"]), rel=1e-5)
