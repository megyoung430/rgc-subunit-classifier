# Scientific methods and assumptions

## Overview

The analysis asks whether spatially coherent filters learned from RGC responses
can serve as receptive-field subunits and improve prediction beyond a single
linear STA filter. Two neural extraction objectives are compared while holding
the preprocessing, spatial crop, subunit selection rule, and frozen-noise
evaluation constant.

## Spike-triggered average

For spike count `c(t, r)` at frame `t` in trial `r` and stimulus history
`S(t-L:t, r)`, the full spatiotemporal STA is:

```text
STA = Σ[r,t] c(t,r) S(t-L:t,r) / Σ[r,t] c(t,r)
```

where only `t >= L` is included and `L = 20` frames. The implementation finds
the largest absolute entry in the `(x, y, lag)` tensor. The spatial STA is the
lag slice through that entry; the temporal STA is the lag course at its spatial
location. Both are normalized to unit Euclidean norm.

## Spatial crop

A correlated 2-D Gaussian is fitted to the spatial STA. The ellipse at `sigma`
Mahalanobis standard deviations defines the mask: every pixel whose centre lies
inside the ellipse, clipped to the image. The crop is the mask's bounding box.
Only masked pixels enter the networks; full-crop weights are reconstructed with
zeros outside the ellipse.

## Temporally filtered ensembles

Each pixel's `L`-frame history is multiplied by the temporal STA and averaged
over lag. This produces one scalar per spatial pixel.

### Classifier data

Every frame `t >= L` contributes an example. Its label is one if the spike count
is nonzero and zero otherwise. Examples are split into training, validation and
test sets by whole trials (25% of trials for testing, 25% of the rest for
validation), because consecutive frames share most of their stimulus history and
a frame-by-frame split would put near-copies of training examples in the test
set. A weighted sampler gives spike and no-spike classes equal expected sampling
frequency during training.

### Autoencoder data

Only frames with nonzero spike count contribute. One frame contributes one
example regardless of spike multiplicity. This reproduces the notebook's
event-triggered ensemble definition. Ensembles are split by whole trials in the
same way as the classifier's examples.

## Classifier

```text
masked pixels → Linear(nodes) → ReLU → Linear(1) → sigmoid probability
```

Binary cross-entropy is optimized with SGD. L1 regularization applies to the
first-layer weights; L2 is optimizer weight decay. An optional step scheduler
reduces the learning rate. Every epoch is run unless `patience` is set, in
which case training stops after that many epochs without an improvement in
validation loss larger than `min_improvement`. The state with best validation
loss is restored. Reported losses are binary cross-entropy without the L1 term,
so they can be compared across regularization settings. The validation loss is
class-balanced (spike and no-spike examples weighted equally), because the
validation set keeps its natural imbalance and plain cross-entropy would pick
the epoch that best predicts "no spike". Setting `patience` without a
validation set is an error.

Trials are assigned to the training, validation and test sets by `split_seed`
(default 0), separately from `random_state`, which sets the network
initialization and sampling. Runs with different `random_state` values are
therefore compared on the same held-out trials.

The STA, temporal filter, crop and mask are computed from all trials before
the split, so the test trials contribute a little to the features the model
sees. Test metrics are slightly optimistic as a result, most noticeably with
few trials.

Test accuracy is not a useful model-quality statistic on its own: spikes are
rare and the held-out set keeps its natural imbalance, so always predicting
"no spike" scores about 95%. The pipeline also reports balanced accuracy (mean
of the two classes' hit rates; 0.5 is chance) and ROC AUC (0.5 is chance).
Inspect these alongside the learned maps, repeated-run stability, and the
derived encoding model's frozen-noise prediction.

## Autoencoder

```text
masked spike-triggered pixels → Linear(nodes) → ReLU
                              → Linear(pixels) → linear (default) or sigmoid reconstruction
```

Mean squared reconstruction error is optimized with Adam. Available penalties
are encoder-weight L1, optimizer L2, and a differentiable neighbor-aware spatial
penalty. Early stopping and best-state restoration work as for the classifier,
and reported losses exclude the penalties. `validation_loss` is the restored
state's validation error; `test_loss` is the error on held-out test trials.

The ensembles are divided by the standard deviation of the training ensembles
(`input_scale`) before training. A temporally filtered ensemble has a mean
square of about 1/L² for an L-lag unit-norm filter (about 0.0025 for 20 lags),
so on the raw scale an encoder L1 penalty of 1e-4 outweighs the whole
reconstruction error and every weight shrinks to zero. On the scaled inputs
predicting zero gives a loss of 1.0, which is the reference line for the
reported losses.

The decoder is linear by default. The temporally filtered ensembles are
zero-centred, so the Colab notebook's sigmoid decoder (outputs between 0 and 1)
cannot reconstruct their negative half; it remains available as
`output_activation="sigmoid"` for comparison.

## Candidate subunit selection

First-layer classifier weights and autoencoder encoder weights are expanded into
the same cropped spatial grid. Moran's I is calculated with four-neighbor
adjacency. A node is selected when:

```text
Moran's I > threshold_morans_i
```

The historical default is 0.25. This is a selection heuristic, not proof that a
filter is a biological subunit. Report the threshold and inspect sensitivity to
it. Do not use the permissive smoke-run threshold (`-1`) for scientific results.

## LN model

The cropped spatiotemporal STA linearly filters each running-noise history. The
filtered values are divided into quantile bins and paired with mean firing rate.
A three-parameter softplus is fit:

```text
f(x) = a1 log(1 + exp(a2 (x + a3)))
```

This model predicts the frozen stimulus without refitting on frozen responses.

## Rectified subunit model

Selected filters are normalized. Least squares finds combination weights that
approximate the cropped spatial STA (with sign corrected for OFF cells). For each
stimulus ensemble, subunit projections are rectified at zero, combined, and
passed through another fitted softplus nonlinearity.

## Held-out comparison

Both LN and subunit models predict frames `L, ..., 299` of the same frozen
stimulus. The observed rate is the repeat-averaged `spk2 / dt`.

- Pearson correlation measures time-course agreement up to affine scaling.
- MSE measures calibrated squared error in Hz².
- AIC/BIC use `n log(MSE)` plus their standard parameter penalties.

AIC/BIC require an explicit parameter-counting convention. In particular,
decide whether STA coefficients and neural spatial filters are counted as fitted
parameters. Correlation and MSE are less ambiguous direct comparisons here.

## Stability across repeated training

`select_stable_layout`:

1. retains runs with the modal subunit count (larger count wins ties);
2. fits an ellipse to every selected subunit;
3. calculates pairwise ellipse Jaccard similarities;
4. uses optimal one-to-one matching between layouts;
5. selects the layout with greatest mean similarity to other eligible runs.

Report the distribution of subunit counts and similarity scores, not only the
selected run.

## Known limitations

- The temporal/spatial STA separation is a slice approximation, not a general
  low-rank decomposition.
- Histories do not cross trial boundaries.
- Moran's I and Gaussian ellipse fits favor smooth localized filters.
- Hyperparameters and thresholds were explored historically; confirm selection
  rules without using frozen-test performance as a tuning target.
- Repeated frozen noise is held-out for prediction, but preprocessing choices and
  exploratory decisions may still create researcher degrees of freedom.
- Neural weights are not uniquely identifiable: permutations, sign/scale
  interactions, and local optima complicate direct node-to-node comparisons.
