# Adapted from stnmf.space.fitellipse in the stnmf package
# (https://github.com/gollischlab/STNMF_with_AFHALS, version 1.1.0).
#
# MIT License
#
# Copyright (c) 2024  Sören J. Zapp and Tim Gollisch
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
"""Fit a 2-D Gaussian to a spatial map and describe it as an ellipse."""

import numpy as np
from scipy.optimize import leastsq
import warnings

__all__ = ["fitellipse"]


def fitellipse(s, sigma=1.0, num_points=100, raw=False, **kwargs):
    """
    Fit a 2d Gaussian to a spatial array and either return coordinate
    data points of an ellipse or its properties

    Parameters
    ----------
    s : (x, y) array_like
            Two dimensional spatial array to which to fit the ellipse

    sigma : float, optional
            Mahalanobis distance (standard deviations from the mean) of the
            ellipse. Default is 1.0

    num_points : int, optional
            Number of coordinate data points to construct the ellipse from.
            Default is 100

    raw : bool, optional
            Return ellipse 'raw' properties instead of data points. Default
            is False

    Returns
    -------
    points : (2, num_points) numpy.ndarray
            Spatial coordinates of the ellipse if `raw=False`

    (mux, muy, d_major, d_minor, deg) : tuple of floats
            Ellipse properties if `raw=True`

    Raises
    ------
    IndexError
            If `s` is not two dimensional.

    ValueError
            If `sigma` is smaller or equal zero.

    Examples
    --------
    The coordinate data points returned if `raw=False` correspond to the
    indices within the array `s` which comprise an ellipse. This is
    useful to use in conjunction with

    >>> from matplotlib import pyplot as plt
    >>> p = fitellipse(s, raw=False)
    >>> plt.imshow(s.T)
    >>> plt.plot(*p)

    The ellipse properties returned if `raw=True` contain values like
    :math:`\\mu_x` and :math:`\\mu_y` and may be used in conjunction
    with

    >>> from matplotlib import pyplot as plt
    >>> from matplotlib import patches
    >>> e = fitellipse(s, raw=True)
    >>> plt.imshow(s.T)
    >>> el = patches.Ellipse((e[0], e[1]), *e[2:], fill=False)
    >>> ax = plt.gca()
    >>> ax.add_artist(el)
    """
    s = np.array(s, copy=True, dtype="float32")
    if s.ndim != 2:
        raise IndexError("s is expected to be two dimensional")
    if sigma <= 0:
        raise ValueError("sigma has to be greater than zero")

    # Fill NaNs
    s[np.isnan(s)] = 0

    def gaussian2d(mux, muy, stdx, stdy, rho, amp):
        def fun(x, y):
            expo = -((2 * (1 - rho**2)) ** -1) * (
                (x - mux) ** 2 / stdx**2
                + (y - muy) ** 2 / stdy**2
                - 2 * rho * (x - mux) * (y - muy) / (stdx * stdy)
            )
            with np.errstate(invalid="ignore"):
                mul = (2 * np.pi * stdx * stdy * np.sqrt(1 - rho**2)) ** -1
            return amp * mul * np.exp(expo)

        return fun

    # Initialize Gaussian parameters around maximum value of s
    max_idx = np.abs(s).argmax()
    max_x, max_y = np.unravel_index(max_idx, s.shape)
    params = np.array([max_x, max_y, 1, 1, 0, 0], dtype="float32")

    # Fit Gaussian
    def lsfun(p):
        return np.ravel(s - gaussian2d(*p)(*np.indices(s.shape)))

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mux, muy, stdx, stdy, rho, amp = leastsq(lsfun, params)[0]

    # Create ellipse from Gaussian
    cov = np.array(
        [[stdx**2, rho * stdx * stdy], [rho * stdx * stdy, stdy**2]]
    )
    eigenval, eigenvec = np.linalg.eigh(cov)
    r_major = np.sqrt(eigenval[1]) * sigma  # Largest eigenvalue
    r_minor = np.sqrt(eigenval[0]) * sigma
    with np.errstate(divide="ignore"):
        rad = np.arctan(eigenvec[1, 1] / eigenvec[0, 1])

    # Return ellipse coordinates instead of data points
    if raw:
        d_major = 2 * r_major
        d_minor = 2 * r_minor
        deg = np.degrees(rad)
        return mux, muy, d_major, d_minor, deg

    # Data points
    ls = np.linspace(0, 2 * np.pi, int(num_points))

    # Calculate position and rotate ellipse
    rot = np.array([[np.cos(rad), np.sin(rad)], [-np.sin(rad), np.cos(rad)]])
    pos = np.array([r_major * np.cos(ls), r_minor * np.sin(ls)])
    points = (pos.T @ rot).T

    # Add x and y offsets
    points[0] += mux
    points[1] += muy

    return points
