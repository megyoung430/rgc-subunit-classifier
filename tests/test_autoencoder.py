import numpy as np
import pytest
import torch

from src.models.subunit_autoencoder import (
    EnsembleDataset,
    SubunitAutoencoder,
    evaluate_autoencoder,
    run_autoencoder,
)


def signed_ensembles(n=200, seed=0):
    rng = np.random.default_rng(seed)
    basis = np.array([[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 1.0, -1.0]])
    return rng.normal(size=(n, 2)) @ basis + 0.05 * rng.normal(size=(n, 4))


def test_default_output_can_reconstruct_signed_inputs():
    torch.manual_seed(0)
    data = EnsembleDataset(signed_ensembles())
    model = SubunitAutoencoder(4, 2)
    assert model.output_activation == "linear"
    model, losses = run_autoencoder(model, data, data, batch_size=50,
                                    learning_rate=0.05, n_epochs=200)
    assert losses["valid"][-1] < 0.1 * np.var(signed_ensembles())
    assert model.subunit_weights.shape == (2, 4)


def test_all_epochs_run_unless_patience_is_given():
    data = EnsembleDataset(signed_ensembles(40))
    _, losses = run_autoencoder(SubunitAutoencoder(4, 2), data, data, batch_size=40,
                                learning_rate=0.0, n_epochs=12)
    assert len(losses["valid"]) == 12
    _, losses = run_autoencoder(SubunitAutoencoder(4, 2), data, data, batch_size=40,
                                learning_rate=0.0, n_epochs=12, patience=2)
    assert len(losses["valid"]) == 3


def test_reported_losses_exclude_the_penalties():
    torch.manual_seed(0)
    values = signed_ensembles(30)
    data = EnsembleDataset(values)
    model, losses = run_autoencoder(SubunitAutoencoder(4, 2), data, data, batch_size=30,
                                    learning_rate=0.0, n_epochs=1, l1_coefficient=10.0)
    with torch.no_grad():
        x = torch.as_tensor(values, dtype=torch.float32)
        mse = torch.nn.functional.mse_loss(model(x), x).item()
    assert losses["valid"][0] == pytest.approx(mse, rel=1e-5)
    assert evaluate_autoencoder(model, data) == pytest.approx(mse, rel=1e-5)


def test_spatial_regularizer_backpropagates():
    values = EnsembleDataset(np.eye(2))
    model = SubunitAutoencoder(2, 1)
    before = model.encoder_layer.weight.detach().clone()
    run_autoencoder(
        model, values, batch_size=2, learning_rate=0.01, n_epochs=1,
        spatial_coefficient=0.1, crop_shape=(1, 2),
        ellipse_mask=np.ones((1, 2), dtype=bool),
    )
    assert not torch.equal(before, model.encoder_layer.weight)


def test_patience_without_validation_data_is_an_error():
    data = EnsembleDataset(signed_ensembles(20))
    with pytest.raises(ValueError, match="patience"):
        run_autoencoder(SubunitAutoencoder(4, 2), data, n_epochs=2, patience=1)


def test_zero_epochs_is_an_error():
    with pytest.raises(ValueError, match="n_epochs"):
        run_autoencoder(SubunitAutoencoder(4, 2), EnsembleDataset(signed_ensembles(20)), n_epochs=0)
