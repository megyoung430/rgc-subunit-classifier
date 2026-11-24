import h5py
import numpy as np
import pytest

from src.simulation.run_pipeline import save_subunits


def test_save_subunits_writes_new_file_and_refuses_to_overwrite(tmp_path):
    path = tmp_path / "Cell 0 subunits.h5"
    save_subunits(path, np.ones((2, 3)))
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["subunits"][:], np.ones((2, 3)))
    with pytest.raises(FileExistsError):
        save_subunits(path, np.zeros((2, 3)))
    with h5py.File(path) as handle:
        np.testing.assert_array_equal(handle["subunits"][:], np.ones((2, 3)))
