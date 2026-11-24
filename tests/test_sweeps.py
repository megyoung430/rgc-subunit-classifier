import csv
import dataclasses
import os
import importlib.util
import shutil
from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import pytest
from scipy.io import savemat

import src.analysis.sweeps as sweeps
from src.analysis.sweeps import (
    MODEL_DEFAULTS,
    SUMMARY_FIELDS,
    completed_run_ids,
    experiment_grid,
    fraction_to_trials,
    parse_values,
    run_identifier,
    run_sweep,
)

from src.simulation.run_autoencoder_pipeline import AutoencoderResult
from src.simulation.run_pipeline import PipelineResult

REPO = Path(__file__).resolve().parents[1]


def tiny_inputs(folder, trials=219):
    folder.mkdir(parents=True, exist_ok=True)
    savemat(folder / "cells.mat", {"spk1": np.zeros((1, 5, trials)), "spk2": np.zeros((1, 5, 2))})
    stim = folder / "stim"
    stim.mkdir(exist_ok=True)
    for trial in range(trials):
        with h5py.File(stim / f"stim{trial:04d}.h5", "w") as handle:
            handle["stim"] = np.ones((4, 5), dtype=np.int8)
    return folder / "cells.mat", stim


def one_config(fraction=0.1, models=("classifier",)):
    return experiment_grid(
        models=list(models), cells=[0], seeds=[0], node_nums=[2],
        l1_coefficients=[0], l2_coefficients=[0], sigmas=[3],
        fractions=[fraction], num_epochs=1,
    )


def test_each_model_gets_its_own_learning_rate_and_batch_size():
    grid = one_config(models=("classifier", "autoencoder"))
    by_model = {config["model"]: config for config in grid}
    assert by_model["classifier"]["learning_rate"] == MODEL_DEFAULTS["classifier"]["learning_rate"]
    assert by_model["classifier"]["batch_size"] == MODEL_DEFAULTS["classifier"]["batch_size"]
    assert by_model["autoencoder"]["learning_rate"] == MODEL_DEFAULTS["autoencoder"]["learning_rate"]
    custom = experiment_grid(
        models=["classifier"], cells=[0], seeds=[0], node_nums=[2],
        l1_coefficients=[0], l2_coefficients=[0], sigmas=[3], fractions=[1.0],
        num_epochs=1, learning_rates={"classifier": [0.1, 0.5]},
        batch_sizes={"classifier": 50},
    )
    assert [c["learning_rate"] for c in custom] == [0.1, 0.5]
    assert {c["batch_size"] for c in custom} == {50}


def test_identifiers_are_order_independent():
    config = one_config()[0]
    assert run_identifier(config) == run_identifier(dict(reversed(list(config.items()))))


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


def test_run_ids_follow_code_and_data_not_their_location(tmp_path, monkeypatch):
    mat, stim = tiny_inputs(tmp_path / "a", trials=3)
    moved = tmp_path / "b"
    shutil.copytree(tmp_path / "a", moved, copy_function=shutil.copy2)
    common = dict(sta_path=None, dry_run=True)
    (first,) = run_sweep(one_config(1.0), data_path=mat, stim_path=stim,
                         output_dir=tmp_path / "out", **common)
    (same,) = run_sweep(one_config(1.0), data_path=moved / "cells.mat", stim_path=moved / "stim",
                        output_dir=tmp_path / "out", **common)
    assert first["run_id"] == same["run_id"]
    monkeypatch.setattr(sweeps, "code_version", lambda: "different code")
    (changed,) = run_sweep(one_config(1.0), data_path=mat, stim_path=stim,
                           output_dir=tmp_path / "out", **common)
    assert changed["run_id"] != first["run_id"]


def test_dry_run_reads_trial_count_without_training(tmp_path):
    mat, stim = tiny_inputs(tmp_path)
    planned = run_sweep(one_config(), data_path=mat, stim_path=stim, sta_path=None,
                        output_dir=tmp_path / "out", dry_run=True)
    assert planned[0]["max_trials"] == 22
    assert not (tmp_path / "out").exists()


def test_dry_run_with_a_cache_folder_writes_nothing(tmp_path):
    mat, stim = tiny_inputs(tmp_path)
    run_sweep(one_config(), data_path=mat, stim_path=stim, sta_path=tmp_path / "cache" / "sta",
              output_dir=tmp_path / "out", cache_dir=tmp_path / "cache", dry_run=True)
    assert not (tmp_path / "cache").exists()
    assert not (tmp_path / "out").exists()


def fake_result(result_type, **values):
    """A real result dataclass with small arrays in the fields the sweep reads."""
    common = dict(
        node_weights=np.zeros((2, 4)), subunits=np.zeros((1, 4)),
        losses={"train": [0.5, 0.4], "valid": [0.6, 0.55]}, ellipse_mask=np.ones((2, 2), bool),
        morans_i=np.array([0.3, 0.1]), split_trials={"test": np.array([1])},
        validation_loss=0.55, test_loss=0.58,
    )
    fields = {field.name: None for field in dataclasses.fields(result_type)}
    unknown = set(common) - set(fields)
    assert not unknown, f"{result_type.__name__} has no fields {unknown}"
    return result_type(**{**fields, **common, **values})


def classifier_result():
    return fake_result(PipelineResult, accuracies={"train": [], "valid": []},
                       test_accuracy=0.9, test_balanced_accuracy=0.7, test_auc=0.8)


def test_summary_rows_report_the_same_measures_for_both_models(tmp_path, monkeypatch):
    mat, stim = tiny_inputs(tmp_path, trials=3)
    monkeypatch.setattr(sweeps, "run_subunit_model", lambda **kw: classifier_result())
    monkeypatch.setattr(sweeps, "run_autoencoder_model",
                        lambda **kw: fake_result(AutoencoderResult, input_scale=1.0))
    rows = run_sweep(one_config(1.0, models=("classifier", "autoencoder")), data_path=mat,
                     stim_path=stim, sta_path=None, output_dir=tmp_path / "out")
    by_model = {row["model"]: row for row in rows}
    for row in rows:
        assert row["status"] == "complete"
        assert row["validation_loss"] == 0.55 and row["test_loss"] == 0.58
    assert by_model["classifier"]["test_balanced_accuracy"] == 0.7
    assert by_model["autoencoder"]["test_balanced_accuracy"] == ""
    with np.load(by_model["classifier"]["artifact"]) as saved:
        np.testing.assert_array_equal(saved["morans_i"], [0.3, 0.1])


def test_summary_from_an_older_format_is_refused(tmp_path):
    mat, stim = tiny_inputs(tmp_path, trials=3)
    (tmp_path / "out").mkdir()
    (tmp_path / "out" / "summary.csv").write_text("run_id,status,primary_loss\n")
    with pytest.raises(ValueError, match="new output folder"):
        run_sweep(one_config(1.0), data_path=mat, stim_path=stim, sta_path=None,
                  output_dir=tmp_path / "out")


def test_existing_artifact_without_complete_row_fails_before_training(tmp_path, monkeypatch):
    def must_not_train(**kwargs):
        raise AssertionError("training started")

    monkeypatch.setattr(sweeps, "run_subunit_model", must_not_train)
    mat, stim = tiny_inputs(tmp_path)
    common = dict(data_path=mat, stim_path=stim, sta_path=None, output_dir=tmp_path / "out")
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
    mat, stim = tiny_inputs(tmp_path, trials=3)
    with pytest.raises(PermissionError):
        run_sweep(one_config(1.0), data_path=mat, stim_path=stim, sta_path=None,
                  output_dir=archive / "sweeps")


@pytest.mark.parametrize("script", ["run_hyperparameter_sweep.py", "run_data_fraction_sweep.py"])
def test_sweep_scripts_exit_non_zero_when_a_run_fails(tmp_path, monkeypatch, script):
    spec = importlib.util.spec_from_file_location("sweep_script", REPO / "scripts" / script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    def broken(**kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(sweeps, "run_subunit_model", broken)
    mat, stim = tiny_inputs(tmp_path, trials=3)
    args = ["--models", "classifier", "--nodes", "2", "--l1", "0", "--seeds", "0",
            "--epochs", "1", "--data", str(mat), "--stimuli", str(stim),
            "--sta", str(tmp_path / "cache" / "sta"), "--cache", str(tmp_path / "cache"),
            "--output", str(tmp_path / "out")]
    if script == "run_data_fraction_sweep.py":
        args += ["--fractions", "1"]
    assert module.main(args) == 1


def test_run_ids_ignore_stimulus_modification_times(tmp_path):
    mat, stim = tiny_inputs(tmp_path, trials=3)
    common = dict(data_path=mat, stim_path=stim, sta_path=None,
                  output_dir=tmp_path / "out", dry_run=True)
    (first,) = run_sweep(one_config(1.0), **common)
    for path in stim.iterdir():
        os.utime(path, (1, 1))
    (again,) = run_sweep(one_config(1.0), **common)
    assert again["run_id"] == first["run_id"]


def test_a_result_that_cannot_be_summarized_leaves_no_artifact(tmp_path, monkeypatch):
    mat, stim = tiny_inputs(tmp_path, trials=3)
    monkeypatch.setattr(sweeps, "run_subunit_model", lambda **kw: classifier_result())
    monkeypatch.setattr(sweeps, "_result_row", lambda result: 1 / 0)
    (row,) = run_sweep(one_config(1.0), data_path=mat, stim_path=stim, sta_path=None,
                       output_dir=tmp_path / "out")
    assert row["status"] == "failed"
    assert not Path(row["artifact"]).exists()
