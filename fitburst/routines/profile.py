from scipy.special import erf
import numpy as np

def compute_profile_gaussian(times: np.ndarray, toa: np.float, width: np.float) -> np.ndarray:
    """
    Computes a one-dimensional Gaussian profile across frequency or time, depending on inputs.

    Parameters
    ----------
    times: np.ndarray
        an array of values corresponding to time or observing frequency

    toa: float
        central value for Gaussian profile

    width: float
        standard deviation (i.e., 'width') of Gaussian profile

    Returns
    -------
    profile: np.ndarray
        an array containing the normalized Gaussian profile
    """

    norm_factor = 1. / np.sqrt(2 * np.pi) / width
    profile = norm_factor * np.exp(-0.5 * ((times - toa) / width)**2)

    return profile

def compute_profile_pbf(times: np.ndarray, toa: np.float, width: np.float, sc_time: np.float) -> np.ndarray:
    """
    Computes a one-dimensional pulse broadening function (PBF) using the analytical 
    solution of a Gaussian profile convolved with a one-sided exponential scattering tail. 

    Parameters
    ----------
    times : np.ndarray
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
    exp_term_2 = np.exp(-(times - toa) / sc_time)
    erf_term = 1 + erf((times - (toa + width**2 / sc_time)) / width / np.sqrt(2))
    profile = amp_term * exp_term_1 * exp_term_2 * erf_term

    return profile
