# Legacy Colab notebook audit

This audit covers `Colab Notebooks/` and `Colab Notebooks-2/`. The directories
contain the same 608 relative files; 607 are byte-identical and only `.DS_Store`
differs. There are 62 notebook copies representing 29 distinct notebook files.

## Incorporated

| Legacy analysis | Project implementation | Notes |
|---|---|---|
| Classifier and autoencoder parameter variants | `src/models/` and `src/simulation/` | Common reproducible pipelines replace Colab globals and duplicated cells. |
| Step learning-rate schedules | `run_model(..., scheduler_step=..., scheduler_gamma=...)` | Present in later classifier notebooks. |
| Repeated-run layout stability | `src.analysis.model_comparison.select_stable_layout` | Uses modal subunit count, optimal one-to-one ellipse matching, and mean Jaccard similarity. |
| MSE, AIC, BIC, and correlation | `src.analysis.model_comparison.prediction_metrics` | Corrects the notebook AIC/BIC sample size from trial count to predicted time-bin count. |
| STNMF comparison | `src.models.stnmf_baseline` | Provides data-shape and execution adapters without copying the bundled third-party source. |
| Subunit and prediction figures | `src.visualization.plotting_fxns` | Returns Matplotlib objects and never changes directories or triggers downloads. |
| Autoencoder neighbour-aware regularization | `src.models.subunit_autoencoder` | Reimplemented with PyTorch operations so gradients propagate. |
| STA/LN and frozen-noise prediction | `src/simulation/generate_training_data.py` and `src/prediction/` | Covers the spatiotemporal STA notebooks' core analysis. |
| Hyperparameter and data-fraction sweeps | `src.analysis.sweeps` and `scripts/run_*_sweep.py` | Reproducible grids with stable IDs, CSV summaries, artifacts, dry runs, and resumption. |

## Already represented before this audit

- Spike-weighted spatiotemporal STA construction, spatial/temporal slices, and
  Gaussian ellipse cropping.
- Binary spike classifier with balanced sampling, L1/L2 regularization, and
  Moran's-I subunit selection.
- Spike-triggered autoencoder ensembles and encoder-weight extraction.
- Softplus LN and rectified subunit models evaluated by frozen-noise Pearson
  correlation.
- Effective receptive-field/subunit diameter and polygon Jaccard functions.

## Useful historical analyses not automatically run

- **Negative-subunit and per-subunit nonlinearity exploration:** the Predictions
  notebook contains extensive cell-specific plotting and several evolving
  attempts. This is potentially useful research, but the versions do not define
  one validated estimator. Existing output PDFs/HDF5 files are retained as
  provenance; no single exploratory attempt was promoted as canonical code.
- **Shared-input experiments:** stored weights and plots investigate constrained
  or shared input structures, but notebook code is tightly cell/file specific and
  does not establish a stable model API. It should become a separate model only
  after its intended constraint and comparison hypothesis are specified.
- **Continuous-time temporal STA:** `Temporal STA Analysis.ipynb` processes a
  different text-file experiment with spike times, frame times, and intensities.
  It is not applicable to the binned `spk1`/`spk2` manuscript dataset, so it was
  not mixed into the current pipeline.

## Intentionally omitted or superseded

- The Sören TensorFlow/Keras autoencoder is omitted because it duplicates the
  PyTorch autoencoder, uses a separate framework, and relies on Colab-only setup.
  Its NNSVD-LRC initialization remains a possible future controlled model variant.
- Copied `STNMF Dependency` and `h5deref Dependency` sources are third-party code.
  They are not vendored into the package; the STNMF adapter uses an installed
  maintained package when requested.
- `regularizer.py` duplicates standard L1, L2, elastic-net, and group-lasso
  formulas. Current models already implement L1/L2 and the autoencoder spatial
  penalty. The untested legacy classes are not copied wholesale.
- Cells that mount Google Drive, install packages, clone repositories, download
  figures, mutate global working directories, or contain fixed `/content/...`
  paths are deployment mechanics rather than research logic.
- `Untitled*`, `Copy of *`, cell-specific NN copies, and regularization notebook
  copies add no distinct validated algorithm beyond the items above.
- Generated PDF, CSV, MAT, HDF5, and model files are results/provenance, not code;
  they were inspected by filename but not treated as implementations.
