# Inferring retinal ganglion cell receptive-field subunits

![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)
![Status](https://img.shields.io/badge/status-research-yellow)
![Tests](https://img.shields.io/badge/tests-15%20passing-brightgreen)

> **Research software under active development.** This repository is a tested
> refactor of manuscript analysis notebooks. Interfaces, defaults, and scientific
> choices may change before publication. Record the commit, environment, random
> seed, and complete model configuration for every reported result.

This project infers spatial subunits within retinal ganglion cell (RGC)
receptive fields from spatiotemporal white-noise responses. It provides two
interchangeable neural extraction methods—a supervised spike classifier and an
unsupervised spike-triggered autoencoder—plus a conventional linear-nonlinear
(LN) baseline and a rectified subunit encoding model. All models can be evaluated
against the same repeated frozen-noise response.

The code began as `Subunit_NN_(Manuscript).py` and several exported Google Colab
notebooks. The maintained implementation is now organized into importable
modules, command-line scripts, interactive tutorials, deterministic tests, and
explicit environment specifications.

## Contents

- [Scientific workflow](#scientific-workflow)
- [Quick start](#quick-start)
- [Data](#data)
- [Tutorials](#tutorials)
- [Running the pipelines](#running-the-pipelines)
- [Experiment sweeps](#experiment-sweeps)
- [Outputs and interpretation](#outputs-and-interpretation)
- [Repository organization](#repository-organization)
- [Reproducibility and behavioral notes](#reproducibility-and-behavioral-notes)
- [Documentation](#documentation)
- [Testing](#testing)
- [Troubleshooting](#troubleshooting)
- [Citation and license](#citation-and-license)

## Scientific workflow

```text
spk1 + running white noise
        │
        ├── spike-weighted spatiotemporal STA
        │       ├── spatial STA → Gaussian ellipse crop
        │       └── temporal STA → temporal filtering
        │
        ├── LN baseline ───────────────────────────────┐
        │                                              │
        ├── classifier (all frames; binary labels)     │
        │       └── hidden weights ─┐                  │
        │                           ├── Moran's I ───┐  │
        └── autoencoder (spike frames only)          │  │
                └── encoder weights ┘                │  │
                                                     │  │
                         selected spatial subunits ──┘  │
                                   │                    │
                         rectified subunit model        │
                                   │                    │
spk2 + repeated frozen noise ◄─────┴────────────────────┘
                         correlation / MSE / AIC / BIC
```

The two neural methods expose the same downstream representation:

- `node_weights`: all hidden/encoder weights, shaped `nodes × cropped pixels`;
- `subunits`: the subset whose spatial Moran's I exceeds the chosen threshold;
- `crop` and `ellipse_mask`: the spatial reference needed to plot those arrays.

See [docs/methods.md](docs/methods.md) for equations, assumptions, and the
differences between the classifier, autoencoder, LN, and subunit models.

## Quick start

Install [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) or
Anaconda, then run from the repository root:

```bash
conda env create -f environment.yaml
conda activate rgc_subunit_classifier
python -m pip install -e .
python -m ipykernel install --user \
  --name rgc_subunit_classifier \
  --display-name "Python (rgc_subunit_classifier)"
python -m pytest
```

For an existing environment:

```bash
conda activate rgc_subunit_classifier
python -m pip install -r requirements.txt
python -m pip install -e .
```

The Conda route is recommended. `environment.yaml` defines Python and the full
analysis/notebook toolchain; `requirements.txt` provides the equivalent pinned
pip packages. Platform-specific GPU builds of PyTorch may require installation
from the [PyTorch selector](https://pytorch.org/get-started/locally/) after the
environment is created.

Detailed setup and update instructions are in
[docs/installation.md](docs/installation.md).

## Data

The expected checked-out layout is:

```text
data/
├── cell_data_01_NC.mat
└── stimulus_data/
    ├── stim0000.h5
    ├── stim0001.h5
    ├── ...
    ├── stim0218.h5
    └── stim_frz.h5
```

The included dataset has:

| Item | Dataset/key | Shape | Meaning |
|---|---|---:|---|
| Running spike counts | `spk1` | `(78, 1500, 219)` | cells × frames × trials |
| Frozen spike counts | `spk2` | `(78, 300, 219)` | cells × frames × repeats |
| Running stimulus | `stim` | `(30000, 1500)` | flattened pixels × frames |
| Frozen stimulus | `stim_frz` | `(200, 150, 300)` | x × y × frames |

All 219 regular stimulus files (`0000`–`0218`) and the frozen stimulus are
currently present. At 30 Hz, the 20-frame filter spans approximately 0.667 s.

See [docs/data.md](docs/data.md) for orientation conventions, units, filename
rules, validation commands, saved STA files, and partial-trial behavior.

## Tutorials

The recommended entry points reproduce the full sequence: fit the LN model,
train one extractor, inspect weights/subunits, fit the derived subunit model,
and compare both encoding models on frozen noise.

| Extractor | Script | Interactive notebook |
|---|---|---|
| Classifier | [demo_classifier_model_comparison.py](scripts/demo_classifier_model_comparison.py) | [demo_classifier_model_comparison.ipynb](scripts/demo_classifier_model_comparison.ipynb) |
| Autoencoder | [demo_autoencoder_model_comparison.py](scripts/demo_autoencoder_model_comparison.py) | [demo_autoencoder_model_comparison.ipynb](scripts/demo_autoencoder_model_comparison.ipynb) |

Full classifier tutorial for cell 0:

```bash
python scripts/demo_classifier_model_comparison.py --cell 0
```

Full autoencoder tutorial:

```bash
python scripts/demo_autoencoder_model_comparison.py --cell 0
```

Open the notebooks with:

```bash
jupyter lab
```

Select the `Python (rgc_subunit_classifier)` kernel. Run notebooks from the
repository root so their relative `data/` and `results/` paths resolve.

### Smoke demonstrations

A smoke demonstration is a deliberately tiny run used only to catch broken
interfaces and runtime errors. Its weights, subunits, and performance must not
be interpreted scientifically.

```bash
python scripts/demo_classifier_model_comparison.py \
  --cell 0 --max-trials 2 --nodes 3 --epochs 1 \
  --batch-size 128 --learning-rate 0.05 --l1 0 \
  --morans-threshold -1 --output /tmp/rgc-classifier-smoke

python scripts/demo_autoencoder_model_comparison.py \
  --cell 0 --max-trials 2 --nodes 3 --epochs 1 \
  --batch-size 128 --learning-rate 0.001 --l1 0 \
  --morans-threshold -1 --output /tmp/rgc-autoencoder-smoke
```

The permissive threshold forces these short runs to reach the subunit-model
stage; it is not a defensible analysis setting.

## Running the pipelines

### Classifier

```python
from src.simulation.run_pipeline import run_subunit_model

result = run_subunit_model(
    cell_num=0,
    data_path="data/cell_data_01_NC.mat",
    stim_path="data/stimulus_data",
    sta_path="results/sta",
    node_num=60,
    learning_rate=0.5,
    num_epochs=100,
    l1_coefficient=1e-4,
    random_state=0,
)

print(result.test_accuracy)
print(result.node_weights.shape, result.subunits.shape)
```

### Autoencoder

```python
from src.simulation.run_autoencoder_pipeline import run_autoencoder_model

result = run_autoencoder_model(
    cell_num=0,
    data_path="data/cell_data_01_NC.mat",
    stim_path="data/stimulus_data",
    sta_path="results/sta",
    node_num=60,
    learning_rate=0.001,
    num_epochs=100,
    l1_coefficient=1e-4,
    output_activation="sigmoid",
    random_state=0,
)

print(result.validation_loss)
print(result.node_weights.shape, result.subunits.shape)
```

All paths are explicit; maintained modules never call `os.chdir`. A non-`None`
`max_trials` selects the first N trial/stimulus pairs and should be reported as a
partial analysis.

## Experiment sweeps

Preview before submitting expensive work:

```bash
python scripts/run_hyperparameter_sweep.py \
  --models classifier --cells 0,3,12 --nodes 7,10,20,40,60 \
  --l1 0,0.0001,0.0005 --learning-rates 0.1,0.5 \
  --seeds 0,1,2 --dry-run
```

Execute a data-fraction sweep:

```bash
python scripts/run_data_fraction_sweep.py \
  --models classifier,autoencoder --cells 0 \
  --fractions 0.1,0.25,0.5,0.75,1 \
  --seeds 0,1,2 --nodes 60 --epochs 100
```

Sweeps write `summary.csv` plus one compressed `.npz` artifact per run. Stable
configuration hashes make runs resumable: completed runs are skipped and failed
runs are retried. Run classifier and autoencoder sweeps separately when their
learning-rate grids differ. See [docs/workflows.md](docs/workflows.md).

## Outputs and interpretation

Tutorial output directories contain:

| File | Contents |
|---|---|
| `summary.json` | Configuration and LN/subunit performance metrics |
| `model_predictions.npz` | Actual rate, time, predictions, nonlinearities, combination weights |
| `*_weights_and_subunits.npz` | Full spatial weights, selected subunits, ellipse mask |
| `*_node_weights.png` | Every hidden/encoder spatial weight |
| `selected_subunits.png` | Moran-selected spatial filters |
| `frozen_noise_predictions.png` | Actual, LN, and subunit-model time series |

Correlation and MSE compare models on identical frozen-noise bins. AIC/BIC are
reported, but conclusions depend on the stated convention for counting learned
spatial filters as fitted parameters. Classifier accuracy and autoencoder
reconstruction loss are different objectives and must not be compared directly.

For repeated runs, use `select_stable_layout` in
`src.analysis.model_comparison` to compare optimally matched ellipse Jaccard
scores among layouts with the modal number of subunits.

## Repository organization

```text
.
├── data/                         spike counts and white-noise stimuli
├── docs/                         installation, data, methods, workflows, audit
├── imported_from_colab/          ignored historical source archive
├── results/                      saved STAs and historical/generated outputs
├── scripts/                      tutorials, smoke run, and sweep entry points
├── src/
│   ├── analysis/                 ellipse, Moran's I, stability, metrics, sweeps
│   ├── models/                   classifier, autoencoder, optional STNMF adapter
│   ├── prediction/               LN and rectified subunit encoding models
│   ├── simulation/               stimulus/STA processing and pipelines
│   └── visualization/            side-effect-free plotting helpers
├── tests/                        deterministic unit and integration tests
├── environment.yaml              recommended Conda environment
├── requirements.txt              pinned pip dependencies
├── pyproject.toml                installable package metadata
└── Subunit_NN_(Manuscript).py    historical Colab export; do not use as API
```

## Reproducibility and behavioral notes

- The filter is `int(0.67 × 30) = 20` frames. The old notebook comment saying
  “600 ms” does not match its calculation.
- Spike counts weight the STA, whereas classifier labels are binary.
- Autoencoder examples include one ensemble per nonzero spike-count frame;
  multiplicity does not duplicate examples, preserving notebook behavior.
- Classifier training uses balanced sampling, stratified splits, and restores
  the best validation-loss weights.
- L1 applies to first-layer/encoder weights; L2 is optimizer weight decay.
- The autoencoder's spatial penalty is differentiable. The notebook version
  detached weights to NumPy and therefore supplied no gradient.
- `output_activation="sigmoid"` reproduces the autoencoder notebook. Because
  filtered ensembles may be negative, `"linear"` is an explicit model variant.
- Crop boundaries are clipped to image bounds rather than allowing negative
  NumPy slices to wrap around the opposite edge.
- Moran's-I threshold, crop sigma, node count, regularization, seed, data
  fraction, environment, and commit should accompany every reported layout.
- CPU and GPU floating-point/order differences can prevent bitwise identity.
  Assess repeated-seed stability rather than relying on a single initialization.

## Documentation

- [Installation and environments](docs/installation.md)
- [Data layout and formats](docs/data.md)
- [Scientific methods and assumptions](docs/methods.md)
- [Tutorials, sweeps, outputs, and recipes](docs/workflows.md)
- [Legacy Colab notebook audit](docs/legacy_notebook_audit.md)

## Testing

```bash
conda activate rgc_subunit_classifier
python -m pytest
```

The suite checks STA weighting and normalization, temporal filtering, ellipse
masking, Moran's I, classifier and autoencoder training, differentiable spatial
regularization, model metrics, layout stability, STNMF shaping, sweep grids,
resumption behavior, and trial-fraction conversion.

## Troubleshooting

- **`ModuleNotFoundError: src`** — run from the repository root and install the
  project with `python -m pip install -e .`.
- **Wrong notebook kernel** — install/select `Python (rgc_subunit_classifier)`.
- **Missing stimulus file** — verify the complete `stim0000.h5`–`stim0218.h5`
  sequence. The default pipeline refuses silent partial analysis.
- **No selected subunits** — do not automatically lower the threshold. Inspect
  all weights, learning curves, tuning, and repeated-seed stability first.
- **PyTorch installation/GPU issue** — use the platform-specific installation
  command from PyTorch; do not mix incompatible CUDA builds.
- **Matplotlib cache warning** — set `MPLCONFIGDIR` to a writable directory.
- **Long runtime** — start with `--dry-run`, a smoke run, or a reported
  `max_trials`; never interpret the reduced run as the final analysis.

More detail is available in [docs/installation.md](docs/installation.md) and
[docs/workflows.md](docs/workflows.md).

## Citation and license

The manuscript citation and software license have not yet been added to this
repository. Before public release, add a `CITATION.cff`, the final paper/data
citations, and an explicit `LICENSE`. Until then, contact the repository owner
before redistribution or reuse beyond normal research collaboration.
