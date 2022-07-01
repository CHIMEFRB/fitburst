"""
Routines for Temporal Shapes of Profiles

This module contains functions that return profile values that are
derivable from analytic expressions. These funtions are used to
derive the temporal variation of the dynamic spectrum. In the case
of pulse broadening (e.g., due to thin-screen scattering), the profile
shape depends on frequency and is thus a function of both time
and frequency.
"""

import scipy.special as ss
import numpy as np

def compute_profile_gaussian(values: float, mean: float, width: float,
                             normalize: bool = False) -> float:
    """
    Computes a one-dimensional Gaussian profile across frequency or time,
    depending on inputs.

    Parameters
    ----------
    values: array_like
        One or more values corresponding to time or observing frequency

    mean: float
        The arithmetic mean value of the Gaussian profile

    width: float
        The standard deviation (i.e., 'width') of the Gaussian profile

    normalize: bool, optional
        If set to True, then apply factor to normalize Gaussian shape

    Returns
    -------
    profile: np.ndarray
        an array containing the normalized Gaussian profile
    """

    norm_factor = 1.

    # if a normalized shape is desired, then apply the width-dependent factor.
    if normalize:
        norm_factor /= np.sqrt(2 * np.pi) / width

    profile = norm_factor * np.exp(-0.5 * ((values - mean) / width)**2)

    return profile

def compute_profile_pbf(times: np.ndarray, toa: np.float, width: np.float,
                        sc_time: np.float) -> np.ndarray:
    """
    Computes a one-dimensional pulse broadening function (PBF) using the
    analytical solution of a Gaussian profile convolved with a one-side
    exponential scattering tail.

    Parameters
    ----------
    times : array_like
        an array of times at which to evaluate PBF

    toa : float
        the time of arrival of the burst

    width : float
        the temporal width of the burst

    sc_time : float
        the scattering timescale quantifying the one-sided exponential tail

    Returns
    -------
    profile: np.ndarray

    Notes
    -----
    The PBF equation is taken from McKinnon et al. (2004).
    """

    amp_term = 1. / 2. / sc_time
    exp_term_1 = np.exp(width**2 / 2 / sc_time**2)
    exp_term_2 = np.exp(-(times - toa) / sc_time[:, None])
    erf_term = 1 + ss.erf((times - (toa + width**2 / sc_time[:, None])) / width / np.sqrt(2))
    profile = amp_term[:, None] * exp_term_1[:, None] * exp_term_2 * erf_term

    return profile
