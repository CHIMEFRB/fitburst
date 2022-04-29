"""
Routines for Spectral Energy Distributions (SEDs)

This module contains functions that return SED values that are
derivable from analytic expression. These funtions are used to
derive the frequency variation of the dynamic spectrum.
"""

import numpy as np

def compute_spectrum_rpl(freqs: np.ndarray, freq_ref: np.float, sp_idx: np.float,
                         sp_run: np.float) -> np.ndarray:
    """
    Computes a one-dimensional frequency spectrum assuming the form of a
    running power law (rpl).

    Parameters
    ----------
    freqs : np.ndarray
        an array of observing frequencies at which to evaluate spectrum

    freq_ref : float
        a reference frequency used for normalization

    sp_idx : float
        spectral index of spectrum

    sp_run : float
        the 'running' of the spectral index, characterizing first-order
        devations from the basic power-law form.

    Returns
    -------
    spectrum : np.ndarray
        the one-dimensional spectrum for input frequencies

    """

    log_freq = np.log(freqs / freq_ref)
    exponent = sp_idx * log_freq + sp_run * log_freq**2
    spectrum = np.exp(exponent)

    return spectrum
