import numpy as np

from src.models.stnmf_baseline import to_stnmf_ensemble


def test_stnmf_adapter_restores_spatial_shape():
    mask = np.array([[True, False], [True, True]])
    compact = np.array([[1, 2, 3], [4, 5, 6]])
    expanded = to_stnmf_ensemble(compact, mask)
    assert expanded.shape == (2, 2, 2)
    np.testing.assert_array_equal(expanded[:, :, 0][mask], compact[0])
    np.testing.assert_array_equal(expanded[:, :, 1][mask], compact[1])
