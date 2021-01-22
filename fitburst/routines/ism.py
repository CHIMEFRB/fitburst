from fitburst.backend import general
import numpy as np

def compute_time_dm_delay(
    dm: np.float, 
    freq1: np.float, 
    freq2: np.float = np.inf, 
    dm_idx: np.float = general["constants"]["index_dispersion"]
    ) -> np.float:
    """
    Computes the time delay due to frequency-dependent dispersion in the ISM.

    Parameters
    ----------
    dm : np.float
        the dispersion measure

    freq1 : np.float
        observing frequency at which to evaluate dispersion delay

    freq2 : np.float, optional
        observing frequency used as relative value

    dm_idx : np.float, optional
        exponent of frequency dependence in dispersion delay

    Returns
    -------
    delay : np.float
        time delay due to dispersion
    """

    dm_constant = general["constants"]["dispersion"]
    delay = dm * dm_constant * (freq1 ** dm_idx - freq2 ** dm_idx)
    
    return delay
    

def compute_time_dm_smear(
    dm: np.float, 
    freq: np.float, 
    bw: np.float, 
    dm_idx: np.float = general["constants"]["index_dispersion"]
    ) -> np.float:
    """
    Computes the time delay due to frequency-dependent dispersion in the ISM.

    Parameters
    ----------
    dm : np.float
        the dispersion measure

    freq : np.float
        observing frequency at which to evaluate smearing timescale

    bw : np.float, optional
        bandwidth of frequency channel

    dm_idx : np.float, optional
        exponent index for frequency dependence of dispersion delay

    Returns
    -------
    smear : np.float
        timescale of dispersion smearing
    """

    dm_constant = general["constants"]["dispersion"]
    smear = 2 * dm * dm_constant * bw * freq ** (dm_idx - 1)

    return smear

def compute_time_scattering(
    freq: np.float, 
    freq_ref: np.float, 
    sc_time_ref: np.float, 
    sc_idx: np.float = general["constants"]["index_scattering"]
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

    sc_idx : np.float, optional
        exponent index for frequency dependence of scattering timescale

    Returns
    -------
    sc_time : np.float
        scattering timescale
    """
    sc_time = sc_time_ref * (freq / freq_ref) ** sc_idx
    return sc_time
