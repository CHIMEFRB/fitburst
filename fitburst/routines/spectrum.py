import numpy as np
from statsmodels.nonparametric.smoothers_lowess import lowess
from scipy.signal import find_peaks
import matplotlib.pyplot as plt 

def compute_spectrum_gaussian(freqs: np.ndarray, freq_ctr: np.float, bw: np.float) -> np.ndarray:
    """
    Computes a one-dimensional frequency spectrum assuming a Gaussian shape.

    Parameters
    ----------
    freqs : np.ndarray
        an array of observing frequencies at which to evaluate spectrum

    freq_ctr : float
        central frequency of emission

    bw : float
        Gaussian width (i.e., bandwidth) of spectrum

    Returns
    -------
    spectrum : np.ndarray
        the one-dimensional spectrum for input frequencies
    """

    norm_factor = 1. / np.sqrt(2 * np.pi) / bw
    spectrum = norm_factor * np.exp(-0.5 * ((freqs - freq_ctr) / bw)**2)
    
    return spectrum

def compute_spectrum_rpl(freqs: np.ndarray, freq_ref: np.float, sp_idx: np.float, sp_run: np.float) -> np.ndarray:
    """
    Computes a one-dimensional frequency spectrum assuming the form of a running power law (rpl).

    Parameters
    ----------
    freqs : np.ndarray
        an array of observing frequencies at which to evaluate spectrum

    freq_ref : float
        a reference frequency used for normalization

    sp_idx : float
        spectral index of spectrum

    sp_run : float
        the 'running' of the spectral index, characterizing first-order devations 
        from the basic power-law form.

    Returns
    -------
    spectrum : np.ndarray
        the one-dimensional spectrum for input frequencies

    """
    
    log_freq = np.log(freqs / freq_ref)
    exponent = -sp_idx * log_freq + sp_run * log_freq**2
    spectrum = np.exp(exponent)

    return spectrum

def rpl(f : np.ndarray, *pars) -> np.ndarray:
    """
    Calculates running power-law model given input parameters and independent variable.
    
    Parameters
    ----------
    f : np.ndarray
        Array of the frequency in MHz of every channel
    pars : np.ndarray
        Array with the parameters of the running power-law model,
        with order amplitude, rpl spectral index, spectral running. 
    Returns
    -------
    np.ndarray
        Running-power law for given parameters evaluated at input f.
    """
    pars = np.array(pars).flatten()
    A, index, running = pars[0], pars[1], pars[2]
    ref_freq = 600.
    return A * (f/ref_freq)**(-index + running * np.log(f/ref_freq))
    
def get_spectrum(power : np.ndarray) -> np.ndarray:
    """
    Computes the normalized spectrum
    
    Parameters
    ----------
    power : np.ndarray
        2D (nfreq, ntime) power
    
    Returns
    -------
    np.ndarray
        Spectrum of the burst 
    """
    return np.sum(power, axis=-1) / np.sqrt(power.shape[-1])
    
def spect_count_components(y : np.ndarray, x : np.ndarray, event_id : str = '',
    diagnostic_plots : bool = False) -> np.ndarray:
    """
    Calculate various parameters needed for lowess smoothing, then perform the smoothing
    and find the number of burst components to fit for.
    
    Parameters
    ----------
    y : np.ndarray
        1D pulse spectrum
    x : float, optional
        Frequency of each bin in MHz
    event_id : str, optional
        Event ID of the FRB
    diagnostic_plots : bool, optional
        If True, displays plots of intermediate steps.
        
    Returns
    -------
    np.ndarray
       Array of peak frequencies found with lowess
    """
    # Use Lowess to smooth the data
    frac = 0.25
    y = np.flip(y)
    smooth = lowess(y,np.flip(x),frac = frac)
    x,y_smooth = smooth[...,0], smooth[...,1]
    # Find peaks 
    peaks = find_peaks(y_smooth, prominence=0, height=0, distance=1)
    peaks, prominences, heights = peaks[0], peaks[1]['prominences'], peaks[1]['peak_heights']
    peaks = np.array([x[p] for p in peaks])
    if len(peaks) == 0:
        peaks = np.array(x[int(len(x)/2)])
        prominences = np.array([2])
    try:
        len(peaks)
    except TypeError:
        peaks = np.array([peaks])
    print(frac)
    print(peaks)
    print(prominences)
    # Prominence and height kwargs added so that find_peaks returns the prominences and heights
    # Sort peaks by prominence
    pos = np.flip(np.argsort(prominences))
    prominences = np.array([prominences[i] for i in pos])
    peaks = np.array([peaks[i] for i in pos])
    print(prominences)
    # Only keep reasonably prominent and high bursts for high SNR bursts:
    keep = prominences > 1
    copy = peaks.copy()
    peaks = peaks[keep]
    if len(peaks) == 0:
        peaks = [copy[0]]
    if diagnostic_plots:
        plt.figure(figsize = (12,10))
        plt.plot(x, y, color = 'k', alpha = 0.3, label = 'Event '+event_id)
        plt.plot(x, y_smooth, color = 'k', label = 'Lowess')
        plt.legend(fontsize = 20)
        for p in peaks:
            plt.axvline(p, color = 'r', ls='--')
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.xlabel('Frequency (MHz)', fontsize = 20)
        plt.show()
    return peaks


