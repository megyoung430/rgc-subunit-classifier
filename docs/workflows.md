# Workflows, tutorials, and outputs

Run all commands from the repository root after activating the environment.

## Recommended analysis order

1. Validate the data layout.
2. Run `python -m pytest`.
3. Run a two-trial smoke demonstration.
4. Run one full tutorial for one cell/seed.
5. Inspect learning curves and all spatial weights.
6. Define hyperparameter selection criteria before a sweep.
7. Run multiple seeds and assess layout stability.
8. Reserve frozen-noise performance for final model comparison.

## Classifier tutorial

```bash
python scripts/demo_classifier_model_comparison.py --help
python scripts/demo_classifier_model_comparison.py \
  --cell 0 --nodes 60 --epochs 100 \
  --learning-rate 0.5 --l1 0.0001 --seed 0
```

Interactive equivalent:

`scripts/demo_classifier_model_comparison.ipynb`

## Autoencoder tutorial

```bash
python scripts/demo_autoencoder_model_comparison.py --help
python scripts/demo_autoencoder_model_comparison.py \
  --cell 0 --nodes 60 --epochs 100 \
  --learning-rate 0.001 --l1 0.0001 --seed 0 \
  --output-activation sigmoid
```

Interactive equivalent:

`scripts/demo_autoencoder_model_comparison.ipynb`

Run sigmoid and linear decoder variants into different output directories to
prevent accidental overwrite.

## Pipeline APIs

Classifier result fields:

| Field | Meaning |
|---|---|
| `model` | Trained `SpikeClassifier` |
| `node_weights` | Expanded first-layer weights |
| `subunits` | Moran-selected weights |
| `sta`, `spatial_sta`, `temporal_sta` | STA components |
| `crop`, `ellipse_mask` | Spatial reference |
| `losses`, `accuracies` | Per-epoch train/validation curves |
| `test_loss`, `test_accuracy` | Held-out classifier metrics |

Autoencoder result fields are analogous, with `validation_loss` and
`n_examples` replacing classifier test metrics.

## Hyperparameter sweeps

Always preview the Cartesian product:

```bash
python scripts/run_hyperparameter_sweep.py \
  --models classifier --cells 0 --seeds 0,1,2 \
  --nodes 7,10,20,40,60 \
  --l1 0,0.0001,0.0005 \
  --learning-rates 0.1,0.5 \
  --epochs 100 --dry-run
```

Remove `--dry-run` to execute. Classifier and autoencoder learning-rate scales
differ, so separate commands are safer than a shared grid.

## Data-fraction sweeps

```bash
python scripts/run_data_fraction_sweep.py \
  --models classifier --cells 0 --seeds 0,1,2 \
  --fractions 0.1,0.25,0.5,0.75,1 \
  --nodes 60 --learning-rates 0.5 --epochs 100
```

Fractions are rounded to a deterministic trial-prefix size. For 219 trials:

| Fraction | Trials |
|---:|---:|
| 0.10 | 22 |
| 0.25 | 55 |
| 0.50 | 110 |
| 0.75 | 164 |
| 1.00 | 219 |

## Sweep outputs and resumption

```text
results/sweeps/<name>/
├── summary.csv
└── artifacts/
    ├── <run-id>.npz
    └── ...
```

The stable run ID hashes model settings, batch size, stopping threshold, and
resolved data paths. A repeated command skips rows marked `complete`. Failed rows
remain retryable. `--fail-fast` stops at the first error; otherwise the sweep
records it and continues.

Artifacts contain spatial weights, selected subunits, ellipse mask, losses, and
available accuracies. The CSV is the experiment index; do not rename artifacts
without updating it.

## Comparing repeated layouts

```python
from src.analysis.model_comparison import select_stable_layout

stable_index, similarity_scores, eligible_indices = select_stable_layout(
    layouts,
    crop_shape=(22, 25),
    sigma=1.5,
)
```

All layouts must use the same crop coordinate system. A score is meaningful only
among runs of the same cell and preprocessing configuration.

## Optional STNMF baseline

```python
from src.models.stnmf_baseline import run_stnmf_baseline

factorization, subunits = run_stnmf_baseline(
    spike_triggered_ensembles,
    ellipse_mask,
    rank=20,
)
```

The external `stnmf` package must be installed. Validate version/API compatibility
against a known historical result before including it in a comparison.

## Output interpretation

- An empty selected-subunit array is a valid outcome, not a software failure.
- Do not lower Moran's-I merely to force a desired number of subunits.
- High classifier accuracy can reflect class imbalance in the natural test set.
- Low autoencoder reconstruction loss does not imply good firing-rate prediction.
- Compare LN and derived subunit predictions on the same frozen frames.
- Use multiple seeds and show the distribution, not only a favorable run.

## Archiving an experiment

Record at minimum:

- Git commit hash and dirty/clean status;
- `environment.yaml` or `conda list --explicit` output;
- cell, model, seed, node count, epochs, batch size, optimizer settings;
- L1/L2/spatial coefficients, crop sigma, Moran threshold;
- number/identity of training trials and whether STA was cached or recomputed;
- sweep summary and artifacts;
- frozen prediction metrics and parameter-counting convention.

Example provenance capture:

```bash
git rev-parse HEAD > results/my_run/git_commit.txt
git status --short > results/my_run/git_status.txt
conda env export -n rgc_subunit_classifier > results/my_run/environment-resolved.yaml
```
