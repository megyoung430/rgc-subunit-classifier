"""Optional adapter for comparing neural models with the historical STNMF baseline."""

import numpy as np


def to_stnmf_ensemble(ensembles, ellipse_mask):
    """Expand compact masked examples into STNMF's x × y × sample format."""
    ensembles = np.asarray(ensembles)
    mask = np.asarray(ellipse_mask, dtype=bool)
    if ensembles.ndim != 2 or ensembles.shape[1] != mask.sum():
        raise ValueError("ensemble width must equal the number of ellipse pixels")
    full = np.zeros((len(ensembles), *mask.shape), dtype=ensembles.dtype)
    full[:, mask] = ensembles
    return np.moveaxis(full, 0, -1)


def run_stnmf_baseline(ensembles, ellipse_mask, rank=20, **kwargs):
    """Run the optional ``stnmf`` package and return full flattened subunits."""
    try:
        import stnmf
    except ImportError as error:
        raise ImportError(
            "STNMF is optional and is not bundled with this project; install "
            "the maintained 'stnmf' package to use this baseline"
        ) from error
    spatial_ensemble = to_stnmf_ensemble(ensembles, ellipse_mask)
    factorization = stnmf.STNMF(spatial_ensemble, r=rank, **kwargs)
    subunits = np.asarray(factorization.subunits).reshape(
        len(factorization.subunits), -1
    )
    return factorization, subunits
