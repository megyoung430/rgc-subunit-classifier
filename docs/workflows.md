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
  --output-activation linear
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
| `losses`, `accuracies` | Per-epoch train/validation curves (validation loss is class-balanced) |
| `validation_loss` | Lowest class-balanced validation loss (the restored weights) |
| `test_loss`, `test_accuracy` | Held-out loss at the natural spike rate (no L1 term) and plain accuracy |
| `test_balanced_loss` | Held-out class-balanced loss, comparable with `validation_loss` (0.693 = chance) |
| `test_balanced_accuracy`, `test_auc` | Class-balanced held-out metrics (0.5 = chance) |
| `split_trials` | Trials in the train, valid and test sets |
| `morans_i` | Moran's I of every node, inside the ellipse mask |

Autoencoder result fields are analogous, with `validation_loss` (restored
state), `test_loss` (held-out trials), `input_scale` and `n_examples` replacing
the classifier's accuracy metrics.

## Hyperparameter sweeps

Always preview the Cartesian product:

```bash
python scripts/run_hyperparameter_sweep.py \
  --models classifier --cells 0 --seeds 0,1,2 \
  --nodes 7,10,20,40,60 \
  --l1 0,0.0001,0.0005 \
  --classifier-learning-rates 0.1,0.5 \
  --epochs 100 --dry-run
```

Remove `--dry-run` to execute. The classifier (SGD) and autoencoder (Adam) need
very different learning rates, so each has its own options:
`--classifier-learning-rates` / `--classifier-batch-size` (defaults 0.5 and 25)
and `--autoencoder-learning-rates` / `--autoencoder-batch-size` (defaults 0.001
and 100). `--patience N` turns on early stopping; by default every epoch runs.

## Data-fraction sweeps

```bash
python scripts/run_data_fraction_sweep.py \
  --models classifier --cells 0 --seeds 0,1,2 \
  --fractions 0.1,0.25,0.5,0.75,1 \
  --nodes 60 --epochs 100
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

The run ID hashes the run's settings (including learning rate, batch size and
patience), the code version (a hash of every Python file under `src/`), the
content of the spike file, and the stimulus files' names and sizes. It does not
depend on where the data are stored or when they were copied. A repeated command skips rows
marked `complete`; failed rows are retried; a re-run after a code change trains
again. `--fail-fast` stops at the first error; otherwise the sweep records it,
continues, and exits with status 1. A `summary.csv` from an older format is
refused: use a new output folder.

Summary columns are the same for both models: `validation_loss` (restored
state; class-balanced cross-entropy for the classifier), `test_loss` (held-out trials; cross-entropy for the classifier,
reconstruction error for the autoencoder), and for the classifier
`test_balanced_loss` (comparable with its `validation_loss`),
`test_balanced_accuracy` and `test_auc`. Artifacts contain spatial weights,
selected subunits, each node's Moran's I, the ellipse mask, losses, available
accuracies and the trials in each split. The CSV is the experiment index; do not rename artifacts
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
