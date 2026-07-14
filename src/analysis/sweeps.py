"""Reproducible, resumable experiment sweeps for both extraction methods."""

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np
from scipy.io import loadmat

from ..simulation.run_autoencoder_pipeline import run_autoencoder_model
from ..simulation.run_pipeline import run_subunit_model


SUMMARY_FIELDS = [
    "run_id", "status", "model", "cell", "seed", "node_num",
    "l1_coefficient", "l2_coefficient", "learning_rate", "sigma",
    "fraction", "max_trials", "num_epochs", "subunit_count",
    "primary_loss", "test_accuracy", "artifact", "error",
]


def parse_values(text, value_type=float):
    """Parse a comma-separated command-line grid value."""
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    if not values:
        raise ValueError("a sweep value list cannot be empty")
    return [value_type(value) for value in values]


def fraction_to_trials(fraction, total_trials):
    """Convert a fraction in (0, 1] into a non-empty prefix of trials."""
    fraction = float(fraction)
    if not 0 < fraction <= 1:
        raise ValueError("data fractions must be greater than 0 and at most 1")
    return max(1, min(total_trials, int(round(fraction * total_trials))))


def experiment_grid(
    *, models, cells, seeds, node_nums, l1_coefficients, l2_coefficients,
    learning_rates, sigmas, fractions, num_epochs,
):
    """Return a deterministic Cartesian product of experiment settings."""
    keys = (
        "model", "cell", "seed", "node_num", "l1_coefficient",
        "l2_coefficient", "learning_rate", "sigma", "fraction",
    )
    values = (
        models, cells, seeds, node_nums, l1_coefficients, l2_coefficients,
        learning_rates, sigmas, fractions,
    )
    return [dict(zip(keys, combination), num_epochs=num_epochs)
            for combination in itertools.product(*values)]


def run_identifier(config):
    """Stable short identifier independent of dictionary insertion order."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def completed_run_ids(summary_path):
    path = Path(summary_path)
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        return {
            row["run_id"] for row in csv.DictReader(handle)
            if row.get("status") == "complete"
        }


def _append_summary(path, row):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in SUMMARY_FIELDS})


def _save_artifact(path, result):
    losses = result.losses
    arrays = {
        "node_weights": result.node_weights,
        "subunits": result.subunits,
        "train_losses": np.asarray(losses.get("train", [])),
        "valid_losses": np.asarray(losses.get("valid", [])),
        "ellipse_mask": result.ellipse_mask,
    }
    if hasattr(result, "accuracies"):
        arrays["train_accuracies"] = np.asarray(result.accuracies.get("train", []))
        arrays["valid_accuracies"] = np.asarray(result.accuracies.get("valid", []))
    np.savez_compressed(path, **arrays)


def run_sweep(
    configs,
    *,
    data_path,
    stim_path,
    sta_path,
    output_dir,
    batch_size,
    stop_threshold=1e-5,
    dry_run=False,
    continue_on_error=True,
):
    """Execute configurations, saving one artifact and one summary row per run.

    Completed run IDs in ``summary.csv`` are skipped, so interrupted sweeps can
    be resumed with the same command. Failed rows are retried on the next run.
    """
    output_dir = Path(output_dir)
    artifact_dir = output_dir / "artifacts"
    summary_path = output_dir / "summary.csv"
    total_trials = loadmat(data_path, variable_names=["spk1"])["spk1"].shape[2]
    completed = completed_run_ids(summary_path)
    planned = []
    for original in configs:
        config = dict(original)
        config["max_trials"] = fraction_to_trials(config["fraction"], total_trials)
        identity = {
            **config, "batch_size": batch_size,
            "stop_threshold": stop_threshold,
            "data_path": str(Path(data_path).resolve()),
            "stim_path": str(Path(stim_path).resolve()),
            "sta_path": str(Path(sta_path).resolve()) if sta_path else None,
        }
        identifier = run_identifier(identity)
        if identifier in completed:
            continue
        planned.append((identifier, config))
    if dry_run:
        return [{"run_id": identifier, **config} for identifier, config in planned]

    artifact_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for identifier, config in planned:
        artifact_path = artifact_dir / f"{identifier}.npz"
        common = dict(
            cell_num=int(config["cell"]), data_path=data_path,
            stim_path=stim_path, sta_path=sta_path,
            node_num=int(config["node_num"]), batch_size=batch_size,
            learning_rate=float(config["learning_rate"]),
            num_epochs=int(config["num_epochs"]),
            stop_threshold=stop_threshold,
            l1_coefficient=float(config["l1_coefficient"]),
            l2_coefficient=float(config["l2_coefficient"]),
            sigma=float(config["sigma"]), random_state=int(config["seed"]),
            max_trials=int(config["max_trials"]),
        )
        row = {"run_id": identifier, **config, "artifact": str(artifact_path)}
        try:
            if config["model"] == "classifier":
                result = run_subunit_model(**common)
                row.update(
                    primary_loss=result.test_loss,
                    test_accuracy=result.test_accuracy,
                )
            elif config["model"] == "autoencoder":
                result = run_autoencoder_model(**common)
                row.update(primary_loss=result.validation_loss, test_accuracy="")
            else:
                raise ValueError(f"unknown model {config['model']!r}")
            _save_artifact(artifact_path, result)
            row.update(status="complete", subunit_count=len(result.subunits), error="")
        except Exception as error:
            row.update(status="failed", subunit_count="", primary_loss="",
                       test_accuracy="", error=f"{type(error).__name__}: {error}")
            _append_summary(summary_path, row)
            rows.append(row)
            if not continue_on_error:
                raise
            continue
        _append_summary(summary_path, row)
        rows.append(row)
    return rows
