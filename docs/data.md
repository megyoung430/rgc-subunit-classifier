# Data layout and formats

## Required files

```text
data/
├── cell_data_01_NC.mat
└── stimulus_data/
    ├── stim0000.h5
    ├── ...
    ├── stim0218.h5
    └── stim_frz.h5
```

Running trial `i` is paired with `stim{i:04d}.h5`; changing filenames or trial
order changes the spike/stimulus correspondence and invalidates the analysis.

## Spike-count MATLAB file

`cell_data_01_NC.mat` contains:

| Key | Shape | Axis order | Description |
|---|---:|---|---|
| `spk1` | `(78, 1500, 219)` | cell, frame, trial | Running-noise spike counts |
| `spk2` | `(78, 300, 219)` | cell, frame, repeat | Frozen-noise spike counts |

Counts may be greater than one. The STA uses multiplicity as a weight. The
classifier converts counts to a binary label (`count != 0`). The autoencoder
uses one example per nonzero frame rather than duplicating it by count.

## Running stimuli

Every `stimNNNN.h5` contains:

| Key | Shape | dtype | Description |
|---|---:|---|---|
| `stim` | `(30000, 1500)` | `int8` | flattened 200 × 150 spatial frame by time |

Stimulus values are expected to be `-1` and `1`. The maintained functions reshape
the flattened dimension to `(x=200, y=150)` using the historical notebook
convention. Do not transpose or change memory order without validating the STA.

## Frozen stimulus

`stim_frz.h5` contains:

| Key | Shape | dtype | Description |
|---|---:|---|---|
| `stim_frz` | `(200, 150, 300)` | `int8` | one frozen sequence repeated for all `spk2` trials |

The actual frozen response is the mean `spk2` count across repeats, divided by
`dt = 1/30 s` to yield Hz.

## STA cache and the Colab archive

Passing `sta_path="results/sta_cache"` (the scripts' default) caches each cell's
STA in one file, `Cell N STA <key>.h5`, with keys `STA` (shape `(30000, 20)`),
`spatial_STA` (`(200, 150)`) and `temp_STA` (`(20,)`). `<key>` is a hash of the
spike counts used, the stimulus size, the filter length, and the stimulus folder
with each trial file's name, size and modification time. A run on a subset of
trials (`max_trials` below 219) or on another stimulus folder therefore computes
and caches its own STA. A cache hit checks the stimulus files' sizes and times
but reads none of them. Cache files are written whole or not at all and never
overwritten; delete `results/sta_cache/` to force a recompute. Omitting
`sta_path` recomputes the STA without caching.

`results/colab_legacy/` holds the outputs of the original Colab analysis
(STAs in `sta/`, subunits, and classifier, STNMF and LN performance tables),
unchanged. The code does not read it: its STAs are in the old three-file format,
so new runs recompute STAs rather than reuse them, and results can differ from
runs that loaded those saved STAs. Every writer in `src/` raises
`PermissionError` inside any folder holding the committed marker file
`.read-only-archive`, so the guard holds in other clones and worktrees too.
Everything else under `results/` is git-ignored.

## Filter and alignment

- Refresh rate: 30 Hz.
- Filter length: `int(0.67 × 30) = 20` frames.
- A sample at frame `t` uses stimulus frames `[t-20, t)`.
- Frames `0`–`19` do not have a complete within-trial history and are excluded.
- The implementation does not concatenate history across trial boundaries.

## Partial analyses

`max_trials=N` uses the deterministic prefix `0, ..., N-1` for both spike counts
and stimulus files. This supports data-fraction sweeps but is not a random subset.
Report `N` or the fraction and consider trial-order effects when interpreting a
learning curve.

By default, the pipelines require all 219 regular files. This prevents missing
data from silently becoming a complete analysis.

## Validation

```bash
conda activate rgc_subunit_classifier
python - <<'PY'
from pathlib import Path
import h5py
from scipy.io import loadmat

root = Path("data/stimulus_data")
spikes = loadmat("data/cell_data_01_NC.mat")
assert spikes["spk1"].shape == (78, 1500, 219)
assert spikes["spk2"].shape == (78, 300, 219)

for trial in range(219):
    path = root / f"stim{trial:04d}.h5"
    assert path.exists(), path
    with h5py.File(path, "r") as handle:
        assert handle["stim"].shape == (30000, 1500)

with h5py.File(root / "stim_frz.h5", "r") as handle:
    assert handle["stim_frz"].shape == (200, 150, 300)
print("Data layout is complete.")
PY
```

## Data provenance

The repository currently does not include a formal public data DOI, license, or
detailed acquisition metadata. Add these before public release. Do not infer
permission to redistribute the biological or stimulus data from its presence in
a working repository.
