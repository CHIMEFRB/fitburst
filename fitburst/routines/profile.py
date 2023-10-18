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

def compute_profile_pbf(time: float, toa: float, width: float, freq: float, ref_freq: float,
                        sc_time_ref: float, sc_index: float = -4.) -> float:
    """
    Computes a one-dimensional pulse broadening function (PBF) using the
    analytical solution of a Gaussian profile convolved with a one-side
    exponential scattering tail.

    Parameters
    ----------
    time : array_like
        a value or array of times at which to evaluate PBF

    toa : float
        the time of arrival of the burst

    width : float
        the temporal width of the burst

    freq : float
        the electromagnetic frequency at which to evaluate the PBF

    ref_freq : float
        the electromagnetic frequency at which to reference the PBF

    sc_time_ref : float
        the scattering timescale at the frequency 'ref_freq'

    sc_index : float, optional
        the exponent of the frequency dependence for the PBF

    Returns
    -------
    profile: np.ndarray

    Notes
    -----
    The PBF equation is taken from McKinnon et al. (2004).
    """

    # evaluate the separate terms that define the PBF.
    amp_term = (freq / ref_freq) ** (-sc_index) 
    sc_time = sc_time_ref * (freq / ref_freq) ** sc_index
    arg_exp = width ** 2 / 2 / sc_time ** 2 - (time - toa) / sc_time
    exp_term = np.exp(arg_exp)
    erf_term = 1 + ss.erf((time - (toa + width ** 2 / sc_time)) / width / np.sqrt(2))

    # finally, compute the PBF.
    profile = amp_term * exp_term * erf_term

    return profile
