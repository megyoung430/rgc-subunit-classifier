import numpy as np
import pytest

from src.simulation.run_autoencoder_pipeline import scale_by_training_std


def test_training_ensembles_have_unit_std_after_scaling():
    rng = np.random.default_rng(0)
    ensembles = rng.normal(scale=0.05, size=(40, 9))
    train = np.arange(30)
    scaled, scale = scale_by_training_std(ensembles, train)
    assert scaled[train].std() == pytest.approx(1.0)
    assert scale == pytest.approx(ensembles[train].std())


def test_held_out_ensembles_do_not_change_the_scale():
    rng = np.random.default_rng(1)
    ensembles = rng.normal(size=(40, 9))
    train = np.arange(30)
    _, before = scale_by_training_std(ensembles, train)
    ensembles[30:] *= 100
    _, after = scale_by_training_std(ensembles, train)
    assert after == before


def test_constant_training_ensembles_are_an_error():
    with pytest.raises(ValueError, match="standard deviation"):
        scale_by_training_std(np.ones((10, 4)), np.arange(5))
