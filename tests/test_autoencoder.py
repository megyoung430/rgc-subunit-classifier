import numpy as np
import torch

from src.models.subunit_autoencoder import (
    EnsembleDataset, SubunitAutoencoder, run_autoencoder,
)


def test_autoencoder_exposes_classifier_compatible_node_weights():
    torch.manual_seed(0)
    values = np.array([
        [-1, -1, 0], [-0.8, -1, 0.1], [1, 1, 0], [0.8, 1, -0.1]
    ], dtype=float)
    dataset = EnsembleDataset(values)
    model = SubunitAutoencoder(3, 2, output_activation="linear")
    model, losses = run_autoencoder(
        model, dataset, dataset, batch_size=4, learning_rate=0.05,
        n_epochs=30, stop_threshold=-1,
    )
    assert model.subunit_weights.shape == (2, 3)
    assert losses["valid"][-1] < losses["valid"][0]


def test_spatial_regularizer_backpropagates():
    values = EnsembleDataset(np.eye(2))
    model = SubunitAutoencoder(2, 1, output_activation="linear")
    before = model.encoder_layer.weight.detach().clone()
    run_autoencoder(
        model, values, batch_size=2, learning_rate=0.01, n_epochs=1,
        spatial_coefficient=0.1, crop_shape=(1, 2),
        ellipse_mask=np.ones((1, 2), dtype=bool),
    )
    assert not torch.equal(before, model.encoder_layer.weight)
