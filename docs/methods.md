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

A correlated 2-D Gaussian is fitted to the spatial STA. An ellipse at `sigma`
Mahalanobis standard deviations defines a rectangular crop and an interior
pixel mask. Crop boundaries are clipped to the image. Only masked pixels enter
the neural models; learned compact weights are expanded back into the rectangle
with zeros outside the ellipse.

## Temporally filtered ensembles

Each pixel's `L`-frame history is multiplied by the temporal STA and averaged
over lag. This produces one scalar per spatial pixel.

### Classifier data

Every frame `t >= L` contributes an example. Its label is one if the spike count
is nonzero and zero otherwise. The train/validation/test splits are stratified.
A weighted sampler gives spike and no-spike classes equal expected sampling
frequency during training.

### Autoencoder data

Only frames with nonzero spike count contribute. One frame contributes one
example regardless of spike multiplicity. This reproduces the notebook's
event-triggered ensemble definition.

## Classifier

```text
masked pixels → Linear(nodes) → ReLU → Linear(1) → sigmoid probability
```

Binary cross-entropy is optimized with SGD. L1 regularization applies to the
first-layer weights; L2 is optimizer weight decay. An optional step scheduler
reduces the learning rate. The state with best validation loss is restored.

Classifier test accuracy is not a sufficient model-quality statistic because
spikes are rare and the held-out set retains its natural imbalance. Inspect loss,
class-sensitive measures if added, learned maps, repeated-run stability, and the
derived encoding model's frozen-noise prediction.

## Autoencoder

```text
masked spike-triggered pixels → Linear(nodes) → ReLU
                              → Linear(pixels) → sigmoid or linear reconstruction
```

Mean squared reconstruction error is optimized with Adam. Available penalties
are encoder-weight L1, optimizer L2, and a differentiable neighbor-aware spatial
penalty. The state with best validation reconstruction loss is restored.

The historical sigmoid decoder is retained. Since temporally filtered inputs can
be negative, a linear decoder is also available and should be evaluated as an
explicit model variant rather than silently replacing the historical behavior.

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
