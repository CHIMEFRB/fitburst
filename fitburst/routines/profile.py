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
                        sc_time_ref: float, sc_index: float = -4., normalize: bool = False) -> float:
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
    sc_time = sc_time_ref * (freq / ref_freq) ** sc_index

    z = (time - toa) / width
    ratio = width / sc_time
    arg1 = (ratio - z) / np.sqrt(2)

    amp_term = np.sqrt(np.pi / 2) * ratio if normalize else sc_time_ref / sc_time

    # Create the output array
    shp = np.broadcast_shapes(freq.shape, time.shape)
    profile = np.empty(shp, dtype=float)

    # Identify regimes where different calculations are needed to prevent numerical overflow
    regime1, regime2 = _identify_pbf_regime(arg1)

    # Calculate the profile in first regime using the scaled complementary error function
    if regime1 is not None:
        profile[..., regime1] = amp_term * np.exp(-0.5 * z[..., regime1] ** 2) * ss.erfcx(arg1[..., regime1])

    # Calculate the profile in second regime using the complementary error function
    if regime2 is not None:
        arg2 = (ratio / 2 - z[..., regime2]) * ratio
        profile[..., regime2] = amp_term * np.exp(arg2) * ss.erfc(arg1[..., regime2])

    return profile


def _identify_pbf_regime(arg: float, threshold=-20.0):
    """Identify times where erfcx suffers from numerical overflow.

    Parameters
    ----------
    arg : np.ndarray[nfreq, ntime] or np.ndarray[ntime,]
        Argument to the scaled complementary error function erfcx.
        In the context of the PBF model, this is given by:
            (width / sc_time - (time - toa) / width) / np.sqrt(2)

    threshold : float, optional
        Values of arg less than this threshold will
        be considered invalid.  Defaults to -20,
        which works well for float64.

    Returns
    -------
    valid : slice, array of int, or None
        Indices in the time axis where the
        scaled complimentary error function
        can be used.  This will be None if
        no time samples qualify.

    invalid : slice, array of int, or None
        Indices in the time axis where the
        scaled complimentary error function
        suffers from numerical overflow.
        This will be None if no time samples
        qualify.
    """
    flag = arg > threshold if arg.ndim == 1 else np.all(arg > threshold, axis=0)
    valid = np.flatnonzero(flag)
    invalid = np.flatnonzero(~flag) if valid.size < flag.size else None

    # If possible, convert from indices to slices so that a copy does not occur
    if (valid[-1] + 1 - valid[0]) == valid.size:
        valid = slice(valid[0], valid[-1]+1)

    if invalid is not None and (invalid[-1] + 1 - invalid[0]) == invalid.size:
        invalid = slice(invalid[0], invalid[-1]+1)

    return valid, invalid
