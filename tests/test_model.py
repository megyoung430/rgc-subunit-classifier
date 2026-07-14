import numpy as np
import pytest

torch = pytest.importorskip("torch")

from src.models.subunit_classifier import SpikeClassifier, SpikeDataset, run_model


def test_model_training_and_testing_smoke():
    torch.manual_seed(0)
    x = np.array([[-1, -1], [-1, 0], [1, 0], [1, 1]], dtype=float)
    y = np.array([0, 0, 1, 1])
    dataset = SpikeDataset(x, y)
    model = SpikeClassifier(2, 3)
    model, losses, accuracies = run_model(
        model, train_set=dataset, valid_set=dataset, batch_size=4,
        learning_rate=0.5, n_epochs=30, stop_thr=-1,
    )
    loss, accuracy = run_model(model, "test", test_set=dataset, batch_size=4)
    assert losses["train"] and accuracies["valid"]
    assert np.isfinite(loss)
    assert accuracy >= 0.75
