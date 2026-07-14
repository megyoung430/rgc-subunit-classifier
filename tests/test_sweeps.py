import csv

import pytest

from src.analysis.sweeps import (
    SUMMARY_FIELDS, completed_run_ids, experiment_grid, fraction_to_trials,
    parse_values, run_identifier, run_sweep,
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


def test_dry_run_reads_trial_count_without_training(tmp_path):
    # Use the real small metadata file; dry-run must create no output artifacts.
    config = experiment_grid(
        models=["classifier"], cells=[0], seeds=[0], node_nums=[2],
        l1_coefficients=[0], l2_coefficients=[0], learning_rates=[0.1],
        sigmas=[3], fractions=[0.1], num_epochs=1,
    )
    planned = run_sweep(
        config, data_path="data/cell_data_01_NC.mat",
        stim_path="data/stimulus_data", sta_path="results/sta",
        output_dir=tmp_path, batch_size=2, dry_run=True,
    )
    assert planned[0]["max_trials"] == 22
    assert not (tmp_path / "summary.csv").exists()
