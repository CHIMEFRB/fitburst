"""
Routines for Dispersion and Scattering Delays

This module contains functions that return temporal quantities
associated with physical delays arising from pulse dispersion
and broadening.
"""

import numpy as np

def compute_time_dm_delay(dm_value: float, dm_const: float, dm_idx: float,
    freq1: float, freq2: float = np.inf) -> float:
    """
    Computes the time delay due to electromagnetic dispersion in the ISM.

    Parameters
    ----------
    dm_value : float
        The dispersion measure

    dm_const : float
        The value of constant applied to DM time delay.

    dm_idx : float
        The exponent of frequency dependence in dispersion delay

    freq1 : float
        The observing frequency at which to evaluate dispersion delay

    freq2 : float, optional
        The observing frequency used as relative value

    Returns
    -------
    delay : float
        time delay due to dispersion
    """

    delay = dm_value * dm_const * (freq1 ** dm_idx - freq2 ** dm_idx)

    return delay


def compute_time_dm_smear(dm_value: float, dm_const: float, dm_idx: float,
    freq: float, width_chan: float) -> float:
    """
    Computes the time delay due to frequency-dependent dispersion in the ISM.

    Parameters
    ----------
    dm_value : float
        The dispersion measure

    dm_const : float
        The value of constant applied to DM time delay.

    dm_idx : float
        The exponent of frequency dependence in dispersion delay

    freq : float
        The observing frequency at which to evaluate smearing timescale

    width_chan : float, optional
        The bandwidth of frequency channel

    Returns
    -------
    smear : float
        timescale of dispersion smearing
    """

    smear = 2 * dm_value * dm_const * width_chan * freq ** (dm_idx - 1)

    return smear

def compute_time_scattering(freq: float, ref_freq: float, sc_time_ref: float,
    sc_idx: float) -> float:
    """
    Computes the scattering timescale as a scaled value relative to scattering determined at
    a reference frequency.

    Parameters
    ----------
    freq : float
        observing frequency at which to evaluate thin-screen scattering

    ref_freq : float
        reference frequency for supplied scattering timescale

    sc_time_ref : float
        scattering timescale measured at supplied reference frequency

    sc_idx : float
        exponent index for frequency dependence of scattering timescale

    Returns
    -------
    sc_time : float
        scattering timescale
    """

    sc_time = sc_time_ref * (freq / ref_freq) ** sc_idx
    return sc_time
