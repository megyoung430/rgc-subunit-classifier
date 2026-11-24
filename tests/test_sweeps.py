import csv

import numpy as np
import pytest
from scipy.io import savemat

from src.analysis.sweeps import (
    SUMMARY_FIELDS,
    completed_run_ids,
    experiment_grid,
    fraction_to_trials,
    parse_values,
    run_identifier,
    run_sweep,
)


def test_grid_is_cartesian_and_identifiers_are_stable():
    grid = experiment_grid(
        models=["classifier", "autoencoder"], cells=[0], seeds=[0, 1],
        node_nums=[7, 10], l1_coefficients=[0.0], l2_coefficients=[0.0],
        learning_rates=[0.1], sigmas=[3.0], fractions=[1.0], num_epochs=2,
    )
    assert len(grid) == 8
    assert run_identifier(grid[0]) == run_identifier(dict(reversed(list(grid[0].items()))))


def test_fraction_conversion_and_cli_parsing():
    assert fraction_to_trials(0.1, 219) == 22
    assert fraction_to_trials(1, 219) == 219
    assert parse_values("7, 10", int) == [7, 10]
    with pytest.raises(ValueError):
        fraction_to_trials(0, 219)


def test_completed_runs_are_resumable(tmp_path):
    path = tmp_path / "summary.csv"
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerow({"run_id": "done", "status": "complete"})
        writer.writerow({"run_id": "retry", "status": "failed"})
    assert completed_run_ids(path) == {"done"}


def tiny_mat(path, trials=219):
    savemat(path, {"spk1": np.zeros((1, 5, trials)), "spk2": np.zeros((1, 5, 2))})
    return path


def one_config(fraction=0.1):
    return experiment_grid(
        models=["classifier"], cells=[0], seeds=[0], node_nums=[2],
        l1_coefficients=[0], l2_coefficients=[0], learning_rates=[0.1],
        sigmas=[3], fractions=[fraction], num_epochs=1,
    )


def test_dry_run_reads_trial_count_without_training(tmp_path):
    planned = run_sweep(
        one_config(), data_path=tiny_mat(tmp_path / "cells.mat"),
        stim_path=tmp_path / "stim", sta_path=None,
        output_dir=tmp_path / "out", batch_size=2, dry_run=True,
    )
    assert planned[0]["max_trials"] == 22
    assert not (tmp_path / "out").exists()


def test_existing_artifact_without_complete_row_fails_before_training(tmp_path, monkeypatch):
    import src.analysis.sweeps as sweeps

    def must_not_train(**kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(sweeps, "run_subunit_model", must_not_train)
    mat = tiny_mat(tmp_path / "cells.mat")
    common = dict(data_path=mat, stim_path=tmp_path / "stim", sta_path=None,
                  output_dir=tmp_path / "out", batch_size=2)
    (planned,) = run_sweep(one_config(), dry_run=True, **common)
    artifact = tmp_path / "out" / "artifacts" / f"{planned['run_id']}.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"left by an interrupted run")
    (row,) = run_sweep(one_config(), **common)
    assert row["status"] == "failed"
    assert "already exists" in row["error"]
    assert artifact.read_bytes() == b"left by an interrupted run"


def test_sweep_output_inside_archive_is_refused(tmp_path, monkeypatch):
    import src.outputs

    archive = tmp_path / "results" / "colab_legacy"
    monkeypatch.setattr(src.outputs, "ARCHIVE", archive)
    config = experiment_grid(
        models=["classifier"], cells=[0], seeds=[0], node_nums=[2],
        l1_coefficients=[0], l2_coefficients=[0], learning_rates=[0.1],
        sigmas=[3], fractions=[1.0], num_epochs=1,
    )
    with pytest.raises(PermissionError):
        run_sweep(
            config, data_path=tiny_mat(tmp_path / "cells.mat"),
            stim_path="data/stimulus_data", sta_path=None,
            output_dir=archive / "sweeps", batch_size=2,
        )
