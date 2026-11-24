"""Reproducible, resumable experiment sweeps for both extraction methods."""

import csv
import hashlib
import itertools
import json
from pathlib import Path

import numpy as np

from ..data_cache import load_spike_data
from ..outputs import check_new_output, check_output_dir, write_new_file
from ..simulation.generate_training_data import stimulus_fingerprint
from ..simulation.run_autoencoder_pipeline import run_autoencoder_model
from ..simulation.run_pipeline import run_subunit_model

# The classifier is trained with SGD and the autoencoder with Adam, so they
# need very different learning rates; these are the settings each was tuned at.
MODEL_DEFAULTS = {
    "classifier": {"learning_rate": 0.5, "batch_size": 25},
    "autoencoder": {"learning_rate": 0.001, "batch_size": 100},
}
SUMMARY_FIELDS = [
    "run_id", "status", "model", "cell", "seed", "node_num",
    "l1_coefficient", "l2_coefficient", "learning_rate", "batch_size", "sigma",
    "fraction", "max_trials", "num_epochs", "subunit_count",
    "validation_loss", "test_loss", "test_balanced_loss", "test_balanced_accuracy", "test_auc",
    "code_version", "artifact", "error",
]
SRC_ROOT = Path(__file__).resolve().parents[1]


def code_version():
    """Short hash of every Python file under ``src/``.

    Part of each run ID, so a sweep re-run after the code changes trains
    again instead of skipping runs that were completed by older code.
    """
    digest = hashlib.sha256()
    for path in sorted(SRC_ROOT.rglob("*.py")):
        digest.update(path.relative_to(SRC_ROOT).as_posix().encode())
        digest.update(path.read_bytes())
    return digest.hexdigest()[:12]


def _file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()[:16]


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
    sigmas, fractions, num_epochs, learning_rates=None, batch_sizes=None,
):
    """Return a deterministic Cartesian product of experiment settings.

    Args:
        learning_rates: Optional ``{model: [rates]}``; models not listed use
            ``MODEL_DEFAULTS``.
        batch_sizes: Optional ``{model: size}``; models not listed use
            ``MODEL_DEFAULTS``.
    """
    learning_rates, batch_sizes = learning_rates or {}, batch_sizes or {}
    unknown = set(models) - set(MODEL_DEFAULTS)
    if unknown:
        raise ValueError(f"unknown model(s) {sorted(unknown)}; choose from {list(MODEL_DEFAULTS)}")
    keys = ("cell", "seed", "node_num", "l1_coefficient", "l2_coefficient", "sigma", "fraction")
    values = (cells, seeds, node_nums, l1_coefficients, l2_coefficients, sigmas, fractions)
    grid = []
    for model in models:
        rates = learning_rates.get(model, [MODEL_DEFAULTS[model]["learning_rate"]])
        batch = batch_sizes.get(model, MODEL_DEFAULTS[model]["batch_size"])
        for rate, combination in itertools.product(rates, itertools.product(*values)):
            grid.append({"model": model, **dict(zip(keys, combination)),
                         "learning_rate": rate, "batch_size": batch,
                         "num_epochs": num_epochs})
    return grid


def run_identifier(config):
    """Stable short identifier independent of dictionary insertion order."""
    encoded = json.dumps(config, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:16]


def completed_run_ids(summary_path):
    """IDs of runs marked complete in ``summary.csv``.

    Raises:
        ValueError: If the file was written in an older summary format.
    """
    path = Path(summary_path)
    if not path.exists():
        return set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != SUMMARY_FIELDS:
            raise ValueError(
                f"{path} was written in an older summary format; use a new output folder"
            )
        return {row["run_id"] for row in reader if row.get("status") == "complete"}


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
        "morans_i": result.morans_i,
        "train_losses": np.asarray(losses.get("train", [])),
        "valid_losses": np.asarray(losses.get("valid", [])),
        "ellipse_mask": result.ellipse_mask,
        **{f"{name}_trials": trials for name, trials in result.split_trials.items()},
    }
    if hasattr(result, "accuracies"):
        arrays["train_accuracies"] = np.asarray(result.accuracies.get("train", []))
        arrays["valid_accuracies"] = np.asarray(result.accuracies.get("valid", []))
    write_new_file(path, lambda temporary: np.savez_compressed(temporary, **arrays))


def _stimulus_identity(stim_path, num_trials):
    """Names and sizes of the stimulus files a run uses.

    Unlike the cache fingerprint, this leaves out the folder and the
    modification times, so copying the data elsewhere keeps the run IDs.
    """
    return [[name, size] for name, size, _ in stimulus_fingerprint(stim_path, num_trials)["files"]]


def _result_row(result):
    """Summary measures, identical in meaning for both models (loss units differ)."""
    return {
        "validation_loss": result.validation_loss,
        "test_loss": result.test_loss,
        "test_balanced_accuracy": getattr(result, "test_balanced_accuracy", ""),
        "test_auc": getattr(result, "test_auc", ""),
        "test_balanced_loss": getattr(result, "test_balanced_loss", ""),
        "subunit_count": len(result.subunits),
    }


def run_sweep(
    configs,
    *,
    data_path,
    stim_path,
    sta_path,
    output_dir,
    patience=None,
    cache_dir=None,
    dry_run=False,
    continue_on_error=True,
):
    """Execute configurations, saving one artifact and one summary row per run.

    Each run's ID hashes its settings, ``patience``, the code version, the
    content of the spike file and the stimulus files' names and sizes (not
    where they are stored or when they were copied). Completed IDs in
    ``summary.csv`` are skipped, so an interrupted sweep resumes with the same
    command; failed rows are retried. ``sta_path`` and ``cache_dir`` are
    caches and do not change results, so they are not part of the ID.

    Returns:
        One row per planned run (with ``dry_run``, the planned configs).
    """
    output_dir = check_output_dir(output_dir)
    artifact_dir = output_dir / "artifacts"
    summary_path = output_dir / "summary.csv"
    # A dry run writes nothing, not even the spike cache.
    spikes_dir = None if cache_dir is None or dry_run else Path(cache_dir) / "spikes"
    total_trials = load_spike_data(data_path, spikes_dir, keys=("spk1",))["spk1"].shape[2]
    completed = completed_run_ids(summary_path)
    version, data_digest = code_version(), _file_digest(data_path)
    planned = []
    for original in configs:
        config = dict(original)
        config["max_trials"] = fraction_to_trials(config["fraction"], total_trials)
        identity = {
            **config, "patience": patience, "code_version": version,
            "spikes": data_digest,
            "stimulus": _stimulus_identity(stim_path, config["max_trials"]),
        }
        identifier = run_identifier(identity)
        if identifier not in completed:
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
            node_num=int(config["node_num"]), batch_size=int(config["batch_size"]),
            learning_rate=float(config["learning_rate"]),
            num_epochs=int(config["num_epochs"]), patience=patience,
            l1_coefficient=float(config["l1_coefficient"]),
            l2_coefficient=float(config["l2_coefficient"]),
            sigma=float(config["sigma"]), random_state=int(config["seed"]),
            max_trials=int(config["max_trials"]), cache_dir=cache_dir,
        )
        row = {"run_id": identifier, **config, "code_version": version,
               "artifact": str(artifact_path)}
        try:
            # An artifact without a "complete" row was left by an interrupted
            # run; fail now rather than retrain and then refuse to overwrite it.
            check_new_output(artifact_path)
            pipeline = {"classifier": run_subunit_model,
                        "autoencoder": run_autoencoder_model}[config["model"]]
            result = pipeline(**common)
            # Build the row first, so a result that cannot be summarized
            # leaves no artifact behind to block the retry.
            measures = _result_row(result)
            _save_artifact(artifact_path, result)
            row.update(status="complete", error="", **measures)
        except Exception as error:
            # Recorded and retried on the next run; the sweep scripts exit
            # non-zero when any row failed.
            row.update(status="failed", error=f"{type(error).__name__}: {error}")
            _append_summary(summary_path, row)
            rows.append(row)
            if not continue_on_error:
                raise
            continue
        _append_summary(summary_path, row)
        rows.append(row)
    return rows


def add_model_arguments(parser):
    """Command-line options for each model's learning rates and batch size."""
    for model, defaults in MODEL_DEFAULTS.items():
        parser.add_argument(f"--{model}-learning-rates", default=str(defaults["learning_rate"]),
                            help=f"comma-separated learning rates for the {model}")
        parser.add_argument(f"--{model}-batch-size", type=int, default=defaults["batch_size"])


def model_settings(args):
    """``(learning_rates, batch_sizes)`` dicts from :func:`add_model_arguments` options."""
    learning_rates = {model: parse_values(getattr(args, f"{model}_learning_rates"))
                      for model in MODEL_DEFAULTS}
    batch_sizes = {model: getattr(args, f"{model}_batch_size") for model in MODEL_DEFAULTS}
    return learning_rates, batch_sizes


def report(rows, dry_run):
    """Print what the sweep did; return the process exit code (1 if any run failed)."""
    print(f"{len(rows)} run(s) {'planned' if dry_run else 'processed'}")
    if dry_run:
        for row in rows:
            print(row)
        return 0
    failed = [row for row in rows if row["status"] == "failed"]
    for row in failed:
        print(f"FAILED {row['run_id']} ({row['model']}, cell {row['cell']}): {row['error']}")
    return 1 if failed else 0
