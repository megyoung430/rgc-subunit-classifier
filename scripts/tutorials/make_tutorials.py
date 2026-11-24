"""Build the three tutorial notebooks (without outputs).

The notebooks in this folder are generated from this file: edit it, then run

    python scripts/tutorials/make_tutorials.py scripts/tutorials

from the repository root, rather than editing the notebooks' JSON by hand.
"""

import sys
from pathlib import Path

import nbformat as nbf

if len(sys.argv) != 2:
    sys.exit("usage: python scripts/tutorials/make_tutorials.py OUTPUT_FOLDER")
OUT = Path(sys.argv[1])

SETUP = '''import os
import sys
from pathlib import Path

# Run from the repository root whichever folder Jupyter starts in, so the
# relative paths below and `import src` work.
REPO = next(p for p in [Path.cwd(), *Path.cwd().parents] if (p / "pyproject.toml").exists())
os.chdir(REPO)
sys.path.insert(0, str(REPO))'''

PATHS = '''# The data folder can live anywhere: set RGC_DATA_ROOT before starting Jupyter
# (e.g. a share or an external drive); otherwise ./data is used.
DATA_ROOT = Path(os.environ.get("RGC_DATA_ROOT", "data"))
DATA_PATH = DATA_ROOT / "cell_data_01_NC.mat"
STIMULUS_PATH = DATA_ROOT / "stimulus_data"
CACHE_DIR = Path("results/cache")          # disposable: delete it to recompute
STA_PATH = CACHE_DIR / "sta"'''

STIMULUS = '''WIDTH, HEIGHT, REFRESH_RATE = 200, 150, 30   # stimulus pixels and frames per second
DT = 1 / REFRESH_RATE
FILTER_LENGTH = int(0.67 * REFRESH_RATE)     # 20 frames ≈ 0.67 s of stimulus history
SIGMA = 3                                    # ellipse size, in standard deviations'''


def md(text):
    return nbf.v4.new_markdown_cell(text.strip("\n"))


def code(text):
    return nbf.v4.new_code_cell(text.strip("\n"))


def notebook(cells):
    nb = nbf.v4.new_notebook()
    nb.cells = cells
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
        "language_info": {"name": "python"},
    }
    return nb


# ---------------------------------------------------------------------------
tutorial_1 = notebook([
    md('''
# Tutorial 1: spikes, stimulus, STA and the LN model

This notebook walks through the first half of the analysis for one cell:

1. **Spikes**: what the recorded spike counts look like.
2. **Stimulus**: the flickering checkerboard the retina saw.
3. **Spike-triggered average (STA)**: the average stimulus before a spike, which shows the cell's receptive field in space and time.
4. **Ellipse crop**: cutting the stimulus down to the receptive field.
5. **Linear-nonlinear (LN) model**: predicting the firing rate from the STA, and testing that prediction on a repeated "frozen" stimulus.

Tutorials 2 and 3 build on this to find the receptive field's **subunits** and test whether a subunit model predicts better than the LN model.

Run the cells in order. Everything slow is cached under `results/cache/`, so a second run takes seconds.
'''),
    md("## Setup"),
    code(SETUP + '''

import h5py
import numpy as np
from matplotlib import pyplot as plt

from src.analysis.model_comparison import correlation_ceiling, prediction_metrics
from src.data_cache import cropped_stimulus, load_spike_data
from src.prediction.sta_model_predictions import (
    binned_response,
    gen_sta_model,
    get_sta_predictions,
    ln_drive,
    nonlin,
    separable_ln_filter,
)
from src.simulation.generate_training_data import crop_sta, gen_sta
from src.visualization.plotting_fxns import (
    MODEL_COLORS,
    OKABE_ITO,
    plot_crop,
    plot_nonlinearity,
    plot_predictions,
    save_figure,
    use_style,
)

use_style()'''),
    code(PATHS + '''
FIGURES = Path("results/tutorials/01_sta_and_ln_model")

CELL = 0
MAX_TRIALS = None   # None uses all 219 trials; e.g. 20 for a quick first look
''' + STIMULUS),
    md('''
## 1. Spikes

`cell_data_01_NC.mat` holds two arrays of spike counts, one count per cell per stimulus frame (1/30 s):

- **`spk1`**: responses to *running* white noise, a different random stimulus on every trial. Shape: cells × frames × trials. This is what every model is fitted to.
- **`spk2`**: responses to *frozen* noise, one short stimulus shown again and again. Shape: cells × frames × repeats. Averaging the repeats gives a clean firing rate to test predictions on; the models never see these responses during fitting.
'''),
    code('''
spikes = load_spike_data(DATA_PATH, CACHE_DIR / "spikes")
spk1, spk2 = spikes["spk1"], spikes["spk2"]
print(f"spk1 (running noise): {spk1.shape}  cells × frames × trials")
print(f"spk2 (frozen noise):  {spk2.shape}  cells × frames × repeats")

num_trials = spk1.shape[2] if MAX_TRIALS is None else min(MAX_TRIALS, spk1.shape[2])
spikecounts = spk1[CELL][:, :num_trials]          # frames × trials for this cell
num_frames = spikecounts.shape[0]
rates = spk1.sum(axis=(1, 2)) / (spk1.shape[1] * spk1.shape[2] * DT)
print(f"Cell {CELL}: {rates[CELL]:.1f} Hz; a spike in {np.mean(spikecounts > 0):.1%} of frames; "
      f"using {num_trials} trials of {num_frames * DT:.0f} s")'''),
    md('''
Spikes are rare: most frames have none. That will matter in Tutorial 2, where a classifier has to tell spike frames from no-spike frames.

Below, left: every cell's mean firing rate, with the chosen cell highlighted. Right: the chosen cell's responses to the frozen noise. Each row of the raster is one repeat of the same stimulus; spikes line up in columns because the cell responds reliably to the same stimulus. The trace underneath is the repeat-averaged firing rate the models will try to predict.
'''),
    code('''
fig, (ax_rates, ax_raster, ax_rate) = plt.subplots(
    1, 3, figsize=(12, 3.2), gridspec_kw={"width_ratios": [1.2, 1.5, 1.5]})
order = np.argsort(rates)[::-1]
colors = [OKABE_ITO["vermillion"] if c == CELL else "0.6" for c in order]
ax_rates.bar(np.arange(len(rates)), rates[order], color=colors, width=1.0)
ax_rates.set(xlabel="Cell (Sorted)", ylabel="Firing Rate (Hz)", title="All Cells")

frozen = spk2[CELL]                                # frames × repeats
time_frozen = np.arange(frozen.shape[0]) * DT
shown = min(40, frozen.shape[1])
rows, cols = np.nonzero(frozen[:, :shown].T)
ax_raster.plot(time_frozen[cols], rows, "|", color="k", markersize=3)
ax_raster.set(xlabel="Time (s)", ylabel="Repeat", title=f"Cell {CELL} Frozen Noise")
ax_rate.plot(time_frozen, frozen.mean(axis=1) / DT, color=MODEL_COLORS["Measured"], lw=1)
ax_rate.set(xlabel="Time (s)", ylabel="Firing Rate (Hz)",
            title=f"Mean Of {frozen.shape[1]} Repeats")
fig.tight_layout()
save_figure(fig, FIGURES / "Firing Rates And Frozen Raster", caption=(
    f"Left: mean firing rate of every cell during running noise, sorted; cell {CELL} in "
    f"vermillion. Middle: spikes of cell {CELL} on the first {shown} repeats of the frozen "
    f"noise. Right: firing rate averaged over all {frozen.shape[1]} repeats."))'''),
    md('''
## 2. Stimulus

Each trial's stimulus is stored in `stimulus_data/stim####.h5` as a pixels × frames array of +1 (white) and −1 (black). Every pixel flips independently at random on every frame: **binary white noise**. Because it contains no structure of its own, any structure in what precedes a spike comes from the cell.

The 30,000 pixel values of each frame unfold into a 200 × 150 image with `reshape(200, 150, frames)`, the layout all the code uses.
'''),
    code('''
with h5py.File(STIMULUS_PATH / "stim0000.h5", "r") as handle:
    trial0 = handle["stim"][:]
print(f"trial 0: {trial0.shape} (pixels × frames), dtype {trial0.dtype}, values {np.unique(trial0)}")
frames = trial0.reshape(WIDTH, HEIGHT, -1)

fig, axes = plt.subplots(1, 3, figsize=(10, 2.8))
for ax, frame in zip(axes, (0, 1, 2)):
    ax.imshow(frames[:, :, frame].T, origin="lower", cmap="gray", vmin=-1, vmax=1)
    ax.set(title=f"Frame {frame} ({frame * DT * 1000:.0f} ms)", xticks=[], yticks=[])
fig.tight_layout()
save_figure(fig, FIGURES / "Stimulus Frames", caption=(
    "Three consecutive frames of the binary white-noise stimulus (trial 0). "
    "Each of the 200 × 150 pixels is independently white (+1) or black (-1) on each frame."))'''),
    md('''
## 3. Spike-triggered average (STA)

For every spike, take the 20 frames (0.67 s) of stimulus that came just before it, and average them, weighting by the number of spikes in the frame. Pixels and moments that tend to drive the cell stand out; everything else averages to zero.

The result is a **spatiotemporal** filter: 30,000 pixels × 20 time lags. Two slices of it are easier to look at:

- the **spatial STA**, the frame at the moment of strongest response (the receptive field's shape);
- the **temporal STA**, the time course at the strongest pixel.

`gen_sta` reads every trial's stimulus once, which is the slowest step; the result is cached under `results/cache/sta/`, keyed on exactly the spikes and stimulus files used, so it is only computed once.
'''),
    code('''
sta, spatial_sta, temporal_sta = gen_sta(
    WIDTH, HEIGHT, spikecounts, CELL, FILTER_LENGTH, num_frames, num_trials,
    sta_path=STA_PATH, stim_path=STIMULUS_PATH,
)
print(f"STA {sta.shape} (pixels × lags), spatial {spatial_sta.shape}, temporal {temporal_sta.shape}")

# Column l of the STA is the frame (FILTER_LENGTH - l) frames before the spike.
lag_times = (np.arange(FILTER_LENGTH) - FILTER_LENGTH) * DT
peak = np.unravel_index(np.abs(spatial_sta).argmax(), spatial_sta.shape)
print(f"Receptive-field peak at pixel {peak}; "
      f"{'ON' if spatial_sta[peak] > 0 else 'OFF'}-like polarity")'''),
    code('''
fig = plt.figure(figsize=(12, 5.5))
grid = fig.add_gridspec(2, 6, height_ratios=[1.4, 1])
ax_space = fig.add_subplot(grid[0, :3])
limit = np.abs(spatial_sta).max()
ax_space.imshow(spatial_sta.T, origin="lower", cmap="bwr", vmin=-limit, vmax=limit)
ax_space.set(title="Spatial STA", xlabel="X (Pixels)", ylabel="Y (Pixels)")
ax_time = fig.add_subplot(grid[0, 3:])
ax_time.axhline(0, color="0.8", lw=0.8)
ax_time.plot(lag_times, temporal_sta, "o-", color=OKABE_ITO["blue"], markersize=3)
ax_time.set(title="Temporal STA", xlabel="Time Before Spike (s)", ylabel="Weight (a.u.)")

cube = sta.reshape(WIDTH, HEIGHT, FILTER_LENGTH)
half = 15
x0, y0 = max(peak[0] - half, 0), max(peak[1] - half, 0)
cube_limit = np.abs(cube).max()
for ax, lag in zip([fig.add_subplot(grid[1, i]) for i in range(6)],
                   np.linspace(FILTER_LENGTH - 1, 4, 6).astype(int)):
    ax.imshow(cube[x0:x0 + 2 * half, y0:y0 + 2 * half, lag].T, origin="lower", cmap="bwr",
              vmin=-cube_limit, vmax=cube_limit)
    ax.set(title=f"{lag_times[lag] * 1000:.0f} ms", xticks=[], yticks=[])
fig.tight_layout()
save_figure(fig, FIGURES / "Spike-Triggered Average", caption=(
    f"Spike-triggered average of cell {CELL} from {num_trials} trials. Top left: spatial STA "
    "at the lag of strongest response (red = white drives the cell, blue = black). Top right: "
    "temporal STA at the peak pixel. Bottom: a 30 × 30 pixel window around the peak at six "
    "lags before the spike."))'''),
    md('''
## 4. Ellipse crop

Almost all of the 30,000 pixels are noise; only the receptive field matters. A 2-D Gaussian is fitted to the spatial STA, and an ellipse drawn `SIGMA` standard deviations from its centre. The **mask** holds every pixel whose centre lies inside the ellipse; the **crop** is the rectangle around it. From here on, every model sees only the pixels inside the mask.
'''),
    code('''
cropped_sta, crop, ellipse_mask = crop_sta(spatial_sta, SIGMA)
crop_shape = cropped_sta.shape
print(f"crop {crop} → {crop_shape[0]} × {crop_shape[1]} pixels; {ellipse_mask.sum()} inside the ellipse")

fig, (ax_full, ax_zoom) = plt.subplots(1, 2, figsize=(10, 3.8))
plot_crop(spatial_sta, crop, ellipse_mask, ax=ax_full)
ax_full.set_title("Spatial STA With Crop")
plot_crop(spatial_sta, crop, ellipse_mask, ax=ax_zoom)
pad = 8
ax_zoom.set(xlim=(crop[0].start - pad, crop[0].stop + pad),
            ylim=(crop[1].start - pad, crop[1].stop + pad), title="Zoom")
fig.tight_layout()
save_figure(fig, FIGURES / "Ellipse Crop", caption=(
    f"Spatial STA of cell {CELL} with the ellipse mask (green, {SIGMA} standard deviations of "
    "the fitted Gaussian) and the crop rectangle (black). Only pixels inside the ellipse "
    "enter the models."))'''),
    md('''
Now cut every trial's stimulus down to the crop. `cropped_stimulus` reads each trial file once and keeps just these pixels, trial first; with `CACHE_DIR` set, the result is saved under `results/cache/stimuli/` and re-opened instantly next time (it is about 150 MB for this cell at 219 trials).
'''),
    code('''
stimuli = cropped_stimulus(
    STIMULUS_PATH, crop, WIDTH, HEIGHT, num_frames, num_trials, CACHE_DIR / "stimuli")
print(f"cropped stimulus: {stimuli.shape} (trials × crop pixels × frames), "
      f"{stimuli.nbytes / 1e6:.0f} MB instead of {num_trials * trial0.nbytes / 1e9:.1f} GB")'''),
    md('''
## 5. The linear-nonlinear (LN) model

The LN model is the standard description of a ganglion cell:

1. **Linear**: filter the recent stimulus with a spatiotemporal filter taken from the STA. The result, the *linear drive*, is one number per frame: how much the stimulus looked like the filter.
2. **Nonlinear**: turn the drive into a firing rate with a fixed curve. Here it is a **softplus**, `a1 · log(1 + exp(a2 · (drive + a3)))`, roughly zero below a threshold and rising above it.

To fit the curve, the drive is split into 40 bins with equal numbers of frames, and the mean firing rate in each bin is plotted against the mean drive. The softplus is fitted to those points.

**Which filter?** The obvious choice is the full cropped STA: one weight for every pixel and every lag. But each of those weights is an average over a limited number of spikes, so every one carries its own noise, and that noise is filtered into the prediction too. A cleaner filter is **separable**: the spatial STA (inside the ellipse mask, zero outside) times the temporal STA. It assumes every pixel has the same time course, which is close to true for a ganglion cell's receptive-field centre, and it replaces thousands of noisy weights with a few hundred.

The separable filter is also the fair baseline for Tutorials 2 and 3: the subunit models there use spatial maps inside the same ellipse with the same temporal STA. With the LN model built the same way, any difference between them comes from the subunits, not from a noisier filter. Below we fit both filters and compare them on held-out data.
'''),
    code('''
# Both filters are crop pixels × lags.
full_sta_filter = sta.reshape(WIDTH, HEIGHT, FILTER_LENGTH)[crop].reshape(cropped_sta.size, FILTER_LENGTH)
ln_filter = separable_ln_filter(spatial_sta, temporal_sta, crop, ellipse_mask)

def fit_softplus(weights):
    return gen_sta_model(
        WIDTH, HEIGHT, spikecounts, crop, *crop_shape, FILTER_LENGTH, num_frames, num_trials,
        weights, STIMULUS_PATH, stimuli=stimuli,
    )

ln_parameters = fit_softplus(ln_filter)
full_sta_parameters = fit_softplus(full_sta_filter)
print("separable LN softplus parameters a1, a2, a3 =", np.round(ln_parameters, 3))

drive = np.concatenate([ln_drive(stimuli[t], ln_filter, FILTER_LENGTH) for t in range(num_trials)])
counts = spikecounts[FILTER_LENGTH:, :].T.ravel()     # the matching spike counts
centers, binned_rates = binned_response(drive, counts, DT)

fig, ax = plt.subplots(figsize=(4.5, 3.4))
plot_nonlinearity(centers, binned_rates, lambda d: nonlin(d, *ln_parameters), ax=ax,
                  color=MODEL_COLORS["LN"])
ax.set_title("LN Output Nonlinearity")
fig.tight_layout()
save_figure(fig, FIGURES / "LN Nonlinearity", caption=(
    "Points: mean firing rate in 40 equal-count bins of the separable LN model's linear "
    "drive (running-noise trials). Line: fitted softplus nonlinearity."))'''),
    md('''
## 6. Testing the LN model on frozen noise

The real test is the frozen noise: a stimulus the model was not fitted to, with a firing rate measured precisely by averaging many repeats. `get_sta_predictions` runs the frozen stimulus through the LN model and returns the prediction alongside the measured rate.

Two numbers summarise the match:

- **Correlation** (Pearson r): does the prediction go up and down at the right times?
- **Mean squared error**: how far off is it, in Hz²?

No model can match the measured rate perfectly, because even the average of many repeats contains some noise. The **noise ceiling** estimates the best correlation any model could reach: split the repeats into two random halves, correlate the two halves' averages, correct for using only half the data (Spearman–Brown), and take the square root. The **normalized correlation** is the model's correlation divided by that ceiling.

With many repeats, the ceiling can be very close to 1: the average of 219 repeats is an almost noise-free estimate of the firing rate, so nearly all of the remaining mismatch is the model's.
'''),
    code('''
def predict_frozen(parameters, weights):
    predictions, _, measured, time = get_sta_predictions(
        CELL, parameters, weights, crop, *crop_shape, DATA_PATH, STIMULUS_PATH,
        filter_length=FILTER_LENGTH, cache_dir=CACHE_DIR,
    )
    return predictions, measured, time

ln_predictions, measured_rate, time = predict_frozen(ln_parameters, ln_filter)
full_sta_predictions, _, _ = predict_frozen(full_sta_parameters, full_sta_filter)
ceiling = correlation_ceiling(spk2[CELL], FILTER_LENGTH)
ln_metrics, full_sta_metrics = (
    prediction_metrics(measured_rate, predictions, FILTER_LENGTH, ceiling=ceiling.max_correlation)
    for predictions in (ln_predictions, full_sta_predictions)
)
print(f"noise ceiling: r = {ceiling.max_correlation:.3f} (reliability {ceiling.reliability:.3f})")
for name, metrics in (("separable LN", ln_metrics), ("full-STA LN", full_sta_metrics)):
    print(f"{name}: r = {metrics.correlation:.3f}, normalized r = "
          f"{metrics.normalized_correlation:.3f}, MSE = {metrics.mse:.1f} Hz²")

fig, ax = plt.subplots(figsize=(10, 3))
plot_predictions(ln_predictions, measured_rate, time, FILTER_LENGTH, label="LN", ax=ax)
ax.set_title(f"Cell {CELL} Frozen Noise, LN r = {ln_metrics.correlation:.2f}")
fig.tight_layout()
save_figure(fig, FIGURES / "LN Frozen Noise Prediction", caption=(
    f"Measured firing rate of cell {CELL} (mean of {spk2.shape[2]} frozen-noise repeats) "
    f"and the separable LN model's prediction. The first {FILTER_LENGTH} frames are dropped "
    f"because the filter needs that much history. r = {ln_metrics.correlation:.3f} (full-STA "
    f"filter: {full_sta_metrics.correlation:.3f}); noise ceiling {ceiling.max_correlation:.3f}."))'''),
    md('''
## Where next

The LN model treats the receptive field as one linear filter. But ganglion cells pool inputs from bipolar cells, and each of those inputs is rectified on its own before they are summed. Those inputs are the receptive field's **subunits**, and a model that rectifies each one separately can capture responses the LN model misses.

- **Tutorial 2** trains a spike classifier and reads candidate subunits from its hidden units.
- **Tutorial 3** finds subunits with an autoencoder instead.

Both then build a subunit model and compare it with this LN model on the same frozen noise.
'''),
])


# ---------------------------------------------------------------------------
def ln_baseline_cells(load_stimuli=True):
    stimuli_code = (
        'stimuli = cropped_stimulus(\n'
        '    STIMULUS_PATH, crop, WIDTH, HEIGHT, num_frames, num_trials, CACHE_DIR / "stimuli")\n'
        if load_stimuli else ""
    )
    return [
        md('''
## The LN baseline

To judge the subunit model, fit the LN model as in Tutorial 1, on the same trials and crop, with the **separable** filter: the spatial STA inside the ellipse times the temporal STA. The subunit model has the same structure (spatial maps inside the ellipse, the same temporal STA), so the only difference between the two is the subunits' rectification. A full-STA LN filter would also be noisier, which would make the subunit model look better for the wrong reason.
'''),
        code(stimuli_code + '''ln_filter = separable_ln_filter(result.spatial_sta, result.temporal_sta, crop,
                                result.ellipse_mask)
ln_parameters = gen_sta_model(
    WIDTH, HEIGHT, spikecounts, crop, *crop_shape, FILTER_LENGTH, num_frames, num_trials,
    ln_filter, STIMULUS_PATH, stimuli=stimuli,
)
ln_predictions, _, measured_rate, time = get_sta_predictions(
    CELL, ln_parameters, ln_filter, crop, *crop_shape, DATA_PATH, STIMULUS_PATH,
    filter_length=FILTER_LENGTH, cache_dir=CACHE_DIR,
)
repeats = load_spike_data(DATA_PATH, CACHE_DIR / "spikes", keys=("spk2",))["spk2"][CELL]
ceiling = correlation_ceiling(repeats, FILTER_LENGTH)
ln_metrics = prediction_metrics(measured_rate, ln_predictions, FILTER_LENGTH,
                                ceiling=ceiling.max_correlation)
print(f"separable LN: r = {ln_metrics.correlation:.3f} (noise ceiling {ceiling.max_correlation:.3f})")'''),
    ]


def guarded(source):
    """Run a cell's code only when the network produced subunits."""
    body = "\n".join(("    " + line) if line.strip() else line for line in source.strip("\n").split("\n"))
    return "\nif HAVE_SUBUNITS:\n" + body


def subunit_model_cells(label, color_key):
    return [
        md(f'''
## The subunit model

The selected subunits become the spatial filters of a **rectified subunit model**:

1. Weight the recent stimulus by the temporal STA and average over time, giving one value per pixel.
2. Pass that through each subunit's spatial filter and **rectify** (set negative values to zero): each subunit responds only to its preferred contrast.
3. Add the rectified subunit outputs with one weight each, and pass the sum through a softplus, as in the LN model.

The combination weights are chosen so that the weighted subunits add up to the cell's spatial STA (least squares; subunits are first scaled to unit size, and the STA's sign is flipped for OFF cells). The softplus is then fitted to the binned drive, as for the LN model.

`gen_subunit_model` returns the fitted softplus parameters, the combination weights and the **normalized** subunits; pass those normalized subunits on to `get_subunit_predictions`.
'''),
        code('''
# With no subunits there is no subunit model; the cells below are then skipped.
HAVE_SUBUNITS = len(result.subunits) > 0
if not HAVE_SUBUNITS:
    print("No node passed the Moran's I threshold, so there is no subunit model to build "
          "and the remaining cells are skipped. The LN result above still stands. Look at "
          "the node maps; check the settings and use all trials before lowering "
          "MORANS_THRESHOLD.")'''),
        code(guarded('''
subunit_parameters, combination_weights, normalized_subunits = gen_subunit_model(
    result.subunits, WIDTH, HEIGHT, spikecounts, FILTER_LENGTH, num_frames, num_trials,
    result.spatial_sta, result.temporal_sta, crop, *crop_shape, STIMULUS_PATH,
    stimuli=stimuli,
)
print("combination weights:", np.round(combination_weights, 3))

drive = np.concatenate([
    subunit_drive(stimuli[t], normalized_subunits, combination_weights, result.temporal_sta,
                  FILTER_LENGTH)
    for t in range(num_trials)
])
counts = spikecounts[FILTER_LENGTH:, :].T.ravel()
centers, binned_rates = binned_response(drive, counts, DT)''')),
        code(guarded(f'''
fig, (ax_weights, ax_curve) = plt.subplots(1, 2, figsize=(9, 3.4))
ax_weights.bar(np.arange(len(combination_weights)), combination_weights,
               color=MODEL_COLORS["{color_key}"])
ax_weights.axhline(0, color="0.6", lw=0.8)
ax_weights.set(xlabel="Subunit", ylabel="Combination Weight", title="Subunit Weights")
plot_nonlinearity(centers, binned_rates, lambda d: nonlin(d, *subunit_parameters),
                  ax=ax_curve, color=MODEL_COLORS["{color_key}"])
ax_curve.set_title("Subunit Model Nonlinearity")
fig.tight_layout()
save_figure(fig, FIGURES / "Subunit Model Fit", caption=(
    "Left: weight of each selected subunit in the subunit model (least-squares fit of the "
    "normalized subunits to the spatial STA). Right: mean firing rate in 40 equal-count bins "
    "of the subunit model's drive (points) and the fitted softplus (line)."))''')),
        md('''
## Comparing the models on frozen noise

Both models now predict the same held-out frozen-noise frames. A subunit model that captures real nonlinear pooling should correlate better with the measured rate than the LN model, and come closer to the noise ceiling.
'''),
        code(guarded(f'''
subunit_predictions, _, _, _ = get_subunit_predictions(
    CELL, subunit_parameters, combination_weights, normalized_subunits, result.temporal_sta,
    crop, *crop_shape, DATA_PATH, STIMULUS_PATH, filter_length=FILTER_LENGTH,
    cache_dir=CACHE_DIR,
)
subunit_metrics = prediction_metrics(measured_rate, subunit_predictions, FILTER_LENGTH,
                                     ceiling=ceiling.max_correlation)
print(f"{{'Model':<22}}{{'r':>7}}{{'normalized r':>14}}{{'MSE (Hz²)':>12}}")
for name, metrics in (("LN", ln_metrics), ("{label}", subunit_metrics)):
    print(f"{{name:<22}}{{metrics.correlation:>7.3f}}{{metrics.normalized_correlation:>14.3f}}"
          f"{{metrics.mse:>12.1f}}")

fig, (ax_trace, ax_bars) = plt.subplots(1, 2, figsize=(12, 3.2),
                                        gridspec_kw={{"width_ratios": [4, 1]}})
plot_predictions(ln_predictions, measured_rate, time, FILTER_LENGTH, label="LN", ax=ax_trace)
ax_trace.plot(time[FILTER_LENGTH:], subunit_predictions, label="{label}", lw=1.2,
              color=MODEL_COLORS["{label}"])
ax_trace.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
ax_trace.set_title(f"Cell {{CELL}} Frozen Noise")
names = ["LN", "{label}"]
ax_bars.bar([0, 1], [ln_metrics.correlation, subunit_metrics.correlation],
            color=[MODEL_COLORS[n] for n in names])
ax_bars.axhline(ceiling.max_correlation, color="0.4", ls="--", lw=1)
ax_bars.text(1.45, ceiling.max_correlation, "Noise\\nCeiling", va="center", fontsize=8)
ax_bars.set(xticks=[0, 1], xticklabels=["LN", "Subunit"], ylabel="Correlation", ylim=(0, 1),
            title="Correlation")
fig.tight_layout()
save_figure(fig, FIGURES / "Frozen Noise Model Comparison", caption=(
    f"Left: measured firing rate of cell {{CELL}} on the frozen noise, with the LN and "
    f"{label.lower()} model predictions. Right: each model's correlation with the measured "
    f"rate; dashed line, noise ceiling (best achievable correlation given the repeats' "
    f"variability). LN r = {{ln_metrics.correlation:.3f}}, subunit r = "
    f"{{subunit_metrics.correlation:.3f}}, ceiling {{ceiling.max_correlation:.3f}}."))''')),
    ]


def node_inspection_cells(kind):
    return [
        md(f'''
## Which nodes are subunits?

Every hidden node has one weight per pixel, so each can be drawn as a map on the crop. Some nodes learn smooth, compact blobs: candidate subunits. Others stay noisy. **Moran's I** scores how smooth a map is: how similar each pixel's weight is to its four neighbours'. It is near 0 for salt-and-pepper noise and close to 1 for a smooth blob, and it is computed on the pixels inside the ellipse only. Nodes scoring above `MORANS_THRESHOLD` are kept as subunits.

The maps below are sorted by Moran's I; the title of each is the node number and its score.
'''),
        code(f'''
order = np.argsort(np.nan_to_num(result.morans_i, nan=-np.inf))[::-1]
titles = [f"{{n}}: I={{result.morans_i[n]:.2f}}" for n in order]
fig, _ = plot_subunits(result.node_weights[order], crop_shape, columns=10, titles=titles,
                       mask=result.ellipse_mask)
fig.suptitle("All {kind} Weights, Sorted By Moran's I", y=1.0)
fig.tight_layout()
save_figure(fig, FIGURES / "All Node Weights", caption=(
    "Spatial weight map of every node ({kind.lower()} weights) on the cropped window (red positive, blue "
    "negative, white zero; grey outline, ellipse mask), sorted by Moran's I computed inside "
    "the mask."))

fig, ax = plt.subplots(figsize=(4.5, 3))
ax.hist(result.morans_i[np.isfinite(result.morans_i)], bins=20, color="0.6")
ax.axvline(MORANS_THRESHOLD, color=OKABE_ITO["vermillion"], lw=1.5)
ax.set(xlabel="Moran's I", ylabel="Nodes",
       title=f"{{len(result.subunits)}} Of {{len(result.morans_i)}} Nodes Selected")
fig.tight_layout()
save_figure(fig, FIGURES / "Morans I Distribution", caption=(
    f"Moran's I of every node; nodes right of the vermillion line (threshold "
    f"{{MORANS_THRESHOLD}}) are kept as subunits."))'''),
        code('''
if len(result.subunits):
    fig, _ = plot_subunits(result.subunits, crop_shape, columns=min(6, len(result.subunits)),
                           mask=result.ellipse_mask)
    fig.suptitle("Selected Subunits", y=1.02)
    fig.tight_layout()
    save_figure(fig, FIGURES / "Selected Subunits", caption=(
        f"The {len(result.subunits)} nodes whose Moran's I exceeds {MORANS_THRESHOLD}."))
    sizes = get_subunit_sizes(result.subunits, *crop_shape)
    print("effective diameter of each positive subunit (µm):", np.round(sizes, 0))
    print(f"receptive field: {get_rf_size(result.spatial_sta[crop]):.0f} µm")'''),
    ]


tutorial_2 = notebook([
    md('''
# Tutorial 2: finding subunits with a spike classifier

Tutorial 1 described the cell with an LN model: one linear filter, then one nonlinearity. Here we look *inside* the receptive field for **subunits**, the separately rectified inputs the cell pools, using a small neural network, and test whether a model built from them predicts the frozen-noise response better.

**The idea.** Train a network to answer a yes/no question for every frame: *will the cell spike?* Its input is the stimulus inside the receptive field; it has one hidden layer of rectifying units (ReLU) and a single output, the spike probability. That architecture is itself a subunit model: each hidden unit filters the stimulus spatially and rectifies, and the output combines them. If training works, some hidden units' input weights take the shape of real subunits.

Steps:

1. Train the classifier (`run_subunit_model`) and check that it learned something.
2. Inspect every hidden unit's spatial weights and select subunits with Moran's I.
3. Build a rectified subunit model from them and compare it with the LN model on frozen noise.

This takes a few minutes per run on all 219 trials.
'''),
    md("## Setup"),
    code(SETUP + '''

import numpy as np
from matplotlib import pyplot as plt

from src.analysis.analyze_subunits import get_rf_size, get_subunit_sizes
from src.analysis.model_comparison import correlation_ceiling, prediction_metrics
from src.data_cache import cropped_stimulus, load_spike_data
from src.prediction.sta_model_predictions import (
    binned_response,
    gen_sta_model,
    get_sta_predictions,
    nonlin,
    separable_ln_filter,
)
from src.prediction.subunit_model_predictions import (
    gen_subunit_model,
    get_subunit_predictions,
    subunit_drive,
)
from src.simulation.run_pipeline import run_subunit_model
from src.visualization.plotting_fxns import (
    MODEL_COLORS,
    OKABE_ITO,
    plot_nonlinearity,
    plot_predictions,
    plot_subunits,
    plot_training_curves,
    save_figure,
    use_style,
)

use_style()'''),
    code(PATHS + '''
FIGURES = Path("results/tutorials/02_classifier_subunit_model")

CELL = 0
MAX_TRIALS = None        # None uses all 219 trials; a small number only checks that the code runs
''' + STIMULUS + '''

# Classifier settings (the values the original analysis used).
NODE_NUM = 60            # hidden units, i.e. candidate subunits
EPOCHS = 100             # passes through the training data
BATCH_SIZE = 25
LEARNING_RATE = 0.5      # SGD
L1_COEFFICIENT = 1e-4    # pushes weights to zero, keeping maps compact
L2_COEFFICIENT = 0
PATIENCE = None          # None runs every epoch; an integer turns on early stopping
MORANS_THRESHOLD = 0.25
SEED = 0'''),
    md('''
## 1. Training the classifier

`run_subunit_model` does everything from Tutorial 1 up to the crop (STA, ellipse, cropped stimulus, all cached), then:

- builds one **example** per frame: the 20 frames of stimulus before it, weighted by the temporal STA and averaged over time, giving one value per pixel inside the ellipse; the **label** is 1 if the frame had a spike;
- splits the examples into training, validation and test sets **by whole trials** (25% of trials for testing, 25% of the rest for validation). Neighbouring frames share most of their stimulus history, so splitting frame by frame would leak near-copies of training examples into the test set;
- trains with **balanced sampling**, drawing spike and no-spike frames equally often, because spikes are rare;
- keeps the weights from the epoch with the lowest validation loss.
'''),
    code('''
result = run_subunit_model(
    CELL, DATA_PATH, STIMULUS_PATH, sta_path=STA_PATH, cache_dir=CACHE_DIR,
    sigma=SIGMA, node_num=NODE_NUM, num_epochs=EPOCHS, batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE, l1_coefficient=L1_COEFFICIENT,
    l2_coefficient=L2_COEFFICIENT, patience=PATIENCE, morans_threshold=MORANS_THRESHOLD,
    random_state=SEED, max_trials=MAX_TRIALS,
)
crop, crop_shape = result.crop, result.ellipse_mask.shape
spikecounts = load_spike_data(DATA_PATH, CACHE_DIR / "spikes", keys=("spk1",))["spk1"][CELL]
num_frames = spikecounts.shape[0]
num_trials = spikecounts.shape[1] if MAX_TRIALS is None else min(MAX_TRIALS, spikecounts.shape[1])
spikecounts = spikecounts[:, :num_trials]
print({name: len(trials) for name, trials in result.split_trials.items()}, "trials per set")'''),
    md('''
**Did it learn?** Plain accuracy is misleading here: a network that never predicts a spike is right on about 95% of frames. Two measures that are not fooled:

- **Balanced accuracy**: the average of the hit rate on spike frames and on no-spike frames. 0.5 is chance.
- **ROC AUC**: the probability that a random spike frame gets a higher spike probability than a random no-spike frame. 0.5 is chance, 1 is perfect.

The loss curves should fall and level off; a validation loss that rises while training loss keeps falling means overfitting (the restored weights are from the lowest validation point).
'''),
    code('''
test_trials = result.split_trials["test"]
spike_fraction = (spikecounts[FILTER_LENGTH:, test_trials] > 0).mean()
print(f"test set: accuracy {result.test_accuracy:.3f} (always 'no spike' would score "
      f"{1 - spike_fraction:.3f})")
print(f"          balanced accuracy {result.test_balanced_accuracy:.3f}, ROC AUC {result.test_auc:.3f} "
      f"(0.5 = chance)")

fig, ax = plt.subplots(figsize=(5, 3.2))
plot_training_curves(result.losses, ax=ax, ylabel="Cross-Entropy")
ax.set_title("Classifier Training")
fig.tight_layout()
save_figure(fig, FIGURES / "Classifier Training Curves", caption=(
    "Binary cross-entropy (without the L1 term) on the training and validation trials per "
    f"epoch. Test set: balanced accuracy {result.test_balanced_accuracy:.3f}, ROC AUC "
    f"{result.test_auc:.3f}."))'''),
    *node_inspection_cells("Classifier Hidden-Node"),
    *ln_baseline_cells(),
    *subunit_model_cells("Classifier Subunit", "Classifier Subunit"),
    md('''
## Things to try

- **Repeat with other seeds** (`SEED`). Networks find somewhat different layouts each run; `src.analysis.model_comparison.select_stable_layout` picks the most typical layout across runs.
- **Vary `NODE_NUM` and `L1_COEFFICIENT`** with `scripts/run_hyperparameter_sweep.py`, which records every run's held-out loss, balanced accuracy and subunit count.
- **Compare with Tutorial 3**, which finds subunits with an autoencoder trained only on spike frames.
'''),
])


tutorial_3 = notebook([
    md('''
# Tutorial 3: finding subunits with an autoencoder

Tutorial 2 found subunits by training a network to *predict* spikes. This tutorial takes an unsupervised route: an **autoencoder** that only ever sees the stimulus just before a spike, and learns to compress and reconstruct it.

**The idea.** Each spike-triggered example (the stimulus before a spike, filtered in time exactly as in Tutorial 2) is squeezed through a small layer of rectifying units and rebuilt from them. To reconstruct well with few units, each unit has to capture a recurring pattern in what drives the cell. Its input (encoder) weights, drawn on the crop, are candidate subunits, and they are selected and used exactly as in Tutorial 2.

Steps:

1. Train the autoencoder (`run_autoencoder_model`) and check its reconstructions.
2. Inspect every unit's spatial weights and select subunits with Moran's I.
3. Build a rectified subunit model and compare it with the LN model on frozen noise.
'''),
    md("## Setup"),
    code(SETUP + '''

import numpy as np
import torch
from matplotlib import pyplot as plt

from src.analysis.analyze_subunits import get_rf_size, get_subunit_sizes
from src.analysis.model_comparison import correlation_ceiling, prediction_metrics
from src.data_cache import cropped_stimulus, load_spike_data
from src.prediction.sta_model_predictions import (
    binned_response,
    gen_sta_model,
    get_sta_predictions,
    nonlin,
    separable_ln_filter,
)
from src.prediction.subunit_model_predictions import (
    gen_subunit_model,
    get_subunit_predictions,
    subunit_drive,
)
from src.simulation.generate_training_data import gen_eses
from src.simulation.run_autoencoder_pipeline import run_autoencoder_model
from src.visualization.plotting_fxns import (
    MODEL_COLORS,
    OKABE_ITO,
    plot_nonlinearity,
    plot_predictions,
    plot_subunits,
    plot_training_curves,
    save_figure,
    use_style,
)

use_style()'''),
    code(PATHS + '''
FIGURES = Path("results/tutorials/03_autoencoder_subunit_model")

CELL = 0
MAX_TRIALS = None        # None uses all 219 trials; a small number only checks that the code runs
''' + STIMULUS + '''

# Autoencoder settings.
NODE_NUM = 60            # encoder units, i.e. candidate subunits
EPOCHS = 100
BATCH_SIZE = 100
LEARNING_RATE = 0.001    # Adam
L1_COEFFICIENT = 0       # see "Settings that matter" below before raising this
L2_COEFFICIENT = 0
SPATIAL_COEFFICIENT = 0  # >0 favours weights whose neighbours are also large
OUTPUT_ACTIVATION = "linear"   # the examples are signed; a sigmoid output cannot rebuild them
PATIENCE = None
MORANS_THRESHOLD = 0.25
SEED = 0'''),
    md('''
## 1. Training the autoencoder

`run_autoencoder_model` computes the STA, crop and cropped stimulus (cached), then builds one example per frame **that contains a spike** (a frame with several spikes still counts once). The examples are split into training, validation and test sets by whole trials, as in Tutorial 2, and the network is trained to minimise the reconstruction error (mean squared error), plus an optional L1 penalty on the encoder weights. The weights from the epoch with the lowest validation error are kept.

**Scaling.** Each example is the stimulus weighted by the temporal STA and averaged over 20 frames. Because the temporal STA has unit length and the pixels are independent noise, the examples are tiny: their mean square is about 1/20² = 0.0025. The pipeline divides every example by the standard deviation of the training examples (`result.input_scale`), so the network works with values of order 1. All losses below are in those scaled units, where always predicting zero gives an error of about **1.0**.

**Settings that matter.** On the unscaled examples, an L1 penalty of 1e-4 on the encoder weights was larger than the whole reconstruction error, so every weight shrank to zero and nothing was learned. Scaling fixes that: the network now reconstructs part of each example. Whether its units become subunits is a separate question. For cell 0 with all 219 trials they did not, with `L1_COEFFICIENT` at 0 or 1e-4: the smoothest node had a Moran's I of only 0.18, so none passed the threshold, while the classifier in Tutorial 2 found 7 compact subunits on the same data. Treat this tutorial as a starting point for that problem, not a finished method. Real subunits should be clearly smaller than the receptive field and sit in different places within it; check both before trusting any layout.

There is no accuracy to check here. Instead, look at the loss curves, the held-out test error, and a few reconstructions.
'''),
    code('''
result = run_autoencoder_model(
    CELL, DATA_PATH, STIMULUS_PATH, sta_path=STA_PATH, cache_dir=CACHE_DIR,
    sigma=SIGMA, node_num=NODE_NUM, num_epochs=EPOCHS, batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE, l1_coefficient=L1_COEFFICIENT,
    l2_coefficient=L2_COEFFICIENT, spatial_coefficient=SPATIAL_COEFFICIENT,
    output_activation=OUTPUT_ACTIVATION, patience=PATIENCE,
    morans_threshold=MORANS_THRESHOLD, random_state=SEED, max_trials=MAX_TRIALS,
)
crop, crop_shape = result.crop, result.ellipse_mask.shape
spikecounts = load_spike_data(DATA_PATH, CACHE_DIR / "spikes", keys=("spk1",))["spk1"][CELL]
num_frames = spikecounts.shape[0]
num_trials = spikecounts.shape[1] if MAX_TRIALS is None else min(MAX_TRIALS, spikecounts.shape[1])
spikecounts = spikecounts[:, :num_trials]
print(f"{result.n_examples} spike-triggered examples; "
      f"{ {name: len(t) for name, t in result.split_trials.items()} } trials per set")
print(f"input scale {result.input_scale:.4f}; validation error {result.validation_loss:.3f}, "
      f"held-out test error {result.test_loss:.3f} (1.0 = predicting zero)")'''),
    md('''
**Reconstructions.** Below, a few held-out examples (top) and the network's reconstructions of them (bottom), drawn on the crop. Good reconstructions keep the coarse light/dark pattern; the fine pixel noise is not expected to come back, because each spike-triggered example is mostly stimulus noise.

The dashed line on the loss plot is the error of always predicting zero: the mean square of the scaled held-out examples, about 1.0. A useful network must beat it. Remember it sees only spike frames (on all 219 trials about 20,600 of them, against about 324,000 labelled frames for the classifier). If the training error falls well below the validation error, the network is fitting the noise in individual examples.
'''),
    code('''
stimuli = cropped_stimulus(
    STIMULUS_PATH, crop, WIDTH, HEIGHT, num_frames, num_trials, CACHE_DIR / "stimuli")
ensembles, ensemble_trials = gen_eses(
    WIDTH, HEIGHT, spikecounts, result.temporal_sta, np.zeros(crop_shape), crop,
    result.ellipse_mask, FILTER_LENGTH, num_frames, num_trials, stimuli=stimuli,
)
# The network was trained on examples divided by result.input_scale.
held_out = ensembles[np.isin(ensemble_trials, result.split_trials["test"])] / result.input_scale
with torch.no_grad():
    device = next(result.model.parameters()).device
    rebuilt = result.model(torch.as_tensor(held_out[:4], dtype=torch.float32, device=device))
    rebuilt = rebuilt.cpu().numpy()

def on_crop(rows):
    full = np.zeros((len(rows), result.ellipse_mask.size))
    full[:, result.ellipse_mask.ravel()] = rows
    return full

fig, axes = plt.subplots(2, 4, figsize=(9, 4.5))
plot_subunits(on_crop(held_out[:4]), crop_shape, axes=axes[0],
              titles=[f"Example {i}" for i in range(4)], mask=result.ellipse_mask)
plot_subunits(on_crop(rebuilt), crop_shape, axes=axes[1],
              titles=[f"Reconstruction {i}" for i in range(4)], mask=result.ellipse_mask)
fig.tight_layout()
save_figure(fig, FIGURES / "Autoencoder Reconstructions", caption=(
    "Top: four held-out spike-triggered examples (stimulus before a spike, weighted by the "
    "temporal STA and averaged over time, inside the ellipse). Bottom: the autoencoder's "
    "reconstructions."))

fig, ax = plt.subplots(figsize=(5, 3.2))
plot_training_curves(result.losses, ax=ax, ylabel="Reconstruction Error (Scaled MSE)")
ax.axhline(np.mean(held_out**2), color="0.5", ls="--", lw=1, label="Predict Zero")
ax.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.01, 1))
ax.set_title("Autoencoder Training")
fig.tight_layout()
save_figure(fig, FIGURES / "Autoencoder Training Curves", caption=(
    "Reconstruction error (without penalties, in units scaled by the training examples' "
    "standard deviation) on the training and validation trials per epoch; dashed, the error "
    f"of always predicting zero on the held-out examples. Held-out test error {result.test_loss:.3f}."))'''),
    *node_inspection_cells("Autoencoder Encoder"),
    *ln_baseline_cells(load_stimuli=False),
    *subunit_model_cells("Autoencoder Subunit", "Autoencoder Subunit"),
    md('''
## Things to try

- **Compare with Tutorial 2** on the same cell: do the classifier and the autoencoder find the same subunits? `src.analysis.model_comparison.layout_similarity` scores how well two layouts overlap.
- **Try `SPATIAL_COEFFICIENT > 0`**, which rewards weights whose neighbours are also large and so favours compact maps.
- **Repeat with other seeds** and pick the most typical layout with `select_stable_layout`.
'''),
])

OUT.mkdir(parents=True, exist_ok=True)
for name, nb in (("01_sta_and_ln_model", tutorial_1),
                 ("02_classifier_subunit_model", tutorial_2),
                 ("03_autoencoder_subunit_model", tutorial_3)):
    nbf.write(nb, OUT / f"{name}.ipynb")
    print("wrote", OUT / f"{name}.ipynb", len(nb.cells), "cells")
