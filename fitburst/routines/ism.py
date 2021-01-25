import numpy as np

def compute_time_dm_delay(
    dm: np.float, 
    dm_const: np.float,
    dm_idx: np.float,
    freq1: np.float, 
    freq2: np.float = np.inf, 
    ) -> np.float:
    """
    Computes the time delay due to frequency-dependent dispersion in the ISM.

    Parameters
    ----------
    dm : np.float
        the dispersion measure

    dm_const : np.float
        the value of constant applied to DM time delay.

    dm_idx : np.float
        exponent of frequency dependence in dispersion delay

    freq1 : np.float
        observing frequency at which to evaluate dispersion delay

    freq2 : np.float, optional
        observing frequency used as relative value

    Returns
    -------
    delay : np.float
        time delay due to dispersion
    """

    delay = dm * dm_const * (freq1 ** dm_idx - freq2 ** dm_idx)
    
    return delay
    

def compute_time_dm_smear(
    dm: np.float, 
    dm_const: np.float,
    dm_idx: np.float,
    freq: np.float, 
    bw: np.float, 
    ) -> np.float:
    """
    Computes the time delay due to frequency-dependent dispersion in the ISM.

    Parameters
    ----------
    dm : np.float
        the dispersion measure

    dm_const : np.float
        the value of constant applied to DM time delay.

    dm_idx : np.float
        exponent of frequency dependence in dispersion delay

    freq : np.float
        observing frequency at which to evaluate smearing timescale

    bw : np.float, optional
        bandwidth of frequency channel

    Returns
    -------
    smear : np.float
        timescale of dispersion smearing
    """

    smear = 2 * dm * dm_const * bw * freq ** (dm_idx - 1)

    return smear

def compute_time_scattering(
    freq: np.float, 
    freq_ref: np.float, 
    sc_time_ref: np.float, 
    sc_idx: np.float
    ) -> np.float:
    """
    Computes the scattering timescale as a scaled value relative to scattering determined at 
    a reference frequency.

    Parameters
    ----------
    freq : np.float
        observing frequency at which to evaluate thin-screen scattering

    freq_ref : np.float
        reference frequency for supplied scattering timescale

    sc_time_ref : np.float
        scattering timescale measured at supplied reference frequency

    sc_idx : np.float
        exponent index for frequency dependence of scattering timescale

    Returns
    -------
    sc_time : np.float
        scattering timescale
    """

    sc_time = sc_time_ref * (freq / freq_ref) ** sc_idx
    return sc_time
