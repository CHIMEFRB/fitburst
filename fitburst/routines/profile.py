from scipy.special import erf, erfc
import numpy as np
from DM_phase import get_dm 
from statsmodels.nonparametric.smoothers_lowess import lowess
import matplotlib.pyplot as plt
import pathlib
import datetime
from scipy.signal import find_peaks

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

def get_signal(profile: np.ndarray, ds: int = 1) -> tuple:
    """
    Estimates the beginning and end of the timeseries that contains the FRB based on S/N and width.

    Parameters
    ----------
    profile : np.ndarray
        The 1D numpy array containing the pulse profile.
    ds : int, optional
        The downsampling fractor of the pulse profile. 
        If not provided, assumed to be 1.

    Returns
    -------
    start : int
        Bin number that marks the beginning of the FRB signal.
    end : int
        Bin number that makrs the end of the FRB signal.
    """
    remaining = profile.copy()
    start = remaining.argmax()
    end = remaining.argmax()
    lims = get_main_peak_lim(remaining)
    start = min(start,lims[0])
    end = max(end, lims[1])
    bins = np.arange(len(profile))
    mask1 = bins < start
    remaining1 = profile[mask1]
    mask2 = bins > end
    remaining2 = profile[mask2]
    #print(start,end)
    f = max(2, 3 / max(1,np.log10(ds+1)))
    if len(remaining1) > max(20, 1000/ds) and len(remaining1[remaining1 > f]) > max(5, 200/ds):
        start1, end1 = get_signal(remaining1)
        if end1 - start1 > max(15, 500/ds):
            print('APPLIED1')
            start = min(start,start1)
            end = max(end, end1)
    if len(remaining2) > max(20, 1000/ds) and len(remaining2[remaining2 > f]) > max(5, 200/ds):
        start2, end2 = get_signal(remaining2)
        if end2 - start2 > max(15, 500/ds):
            print('APPLIED2')
            start = min(start,start2+end)
            end = max(end, end2+end)
    d = end - start
    if max(profile) < 20:
        add = round(d / 1.2)
    else:
        add = round(d / 1.5)
    if start - add >= 0 and end + add < len(profile):
        start -= add
        end += add
    else:
        if start - add/2 >= 0 and end + add < len(profile):
            start -= add/2
            end += add
        elif end + add/2 < len(profile) and start - add >= 0:
            end += add/2
            start -= add
        else:
            add = round(min(start, len(profile) - end - 1))
            if start - add >= 0 and end + add < len(profile):
                start -= add
                end += add
            elif end + add < len(profile) and start - add >= 0:
                end += add
                start -= add
            else:
                start = 0
                end = len(profile) - 1
    #print(start,end)
    return int(start), int(end)
    
def get_structure_max_DM(wfall: np.ndarray, freq: np.ndarray, DM_range: float = 3., 
    t_res: float = 2.56e-6, ref_freq: str = "bottom", diagnostic_plots: bool = True) -> tuple:
    """
    Applies structure maximising DM correction using the DM_phase module.

    Parameters
    ----------
    wfall : np.ndarray
        2D numpy array containing the spectrum / waterfall (S/N array with shape (nfreqs, ntime)).
    freq : np.ndarray
        1D array of the frequency in MHz of each frequency channel in the waterfall.
	len(freq) should be = wfall.shape[0]
    DM_range : float, optional
        The structure-maximising DM correction will be searched for in this range.
        i.e. the best DM is in [DM0 - DM_range/2., DM0 + DM_range/2.], where DM0 is DM of wfall.
    t_res : float, optional
        Time resolution of the waterfall data.
    ref_freq : str, optional
        Reference frequency is "bottom" (400 MHz), "top" (800 MHz), or "center" (600 MHz).
    diagnostic_plots : bool, optional
        Plot diagnostic plots from DM_phase de-dispersion (True), no plots (False).

    Returns
    -------
    opt_dm : float 
        Best structure-maximising DM correction.
    opt_dm_e : float
        One sigma error on the structure-maximising DM correction. 
    """
    if len(wfall.shape) > 2:
        snr = []
        for i in range(wfall.shape[1]):
            snr.append(np.nanmax(get_profile(wfall[:,i,:])))
        #Pick brightest beam:
        wfall = wfall[:,np.where(snr == np.nanmax(snr))[0][0],:]
    trials = 256
    dms = np.linspace(- DM_range/2., DM_range/2., trials)
    # Flip waterfall and freq array so that diagnostic plots show frequencies in the right order
    # i.e. without this, diagnostic plot waterfall is 800 - 400 MHz.
    if freq[0] > freq[-1]:
        freq = np.flip(freq)
        wfall = np.flip(wfall, axis = 0)
    opt_dm, opt_dm_e = get_dm(wfall,dms, t_res,freq,blackonwhite=True, 
                            manual_cutoff=False, manual_bandwidth=False,ref_freq=ref_freq, no_plots= not diagnostic_plots)
    return opt_dm, opt_dm_e

def exponorm(x: np.ndarray, lam : float, mu: float, sigma: float) -> np.ndarray:
    """
    Calculate the exponentially modified gaussian (EMG) for the given
    input x and parameters lam, mu, and sigma.
    
    Parameters
    ----------
    x : np.ndarray
        Time in s at every time bin
    lam : float
        Inverse of the exponent relaxation time
    mu : float
        Gaussian mean
    sigma : float
        Gaussian std. For more info, see:
        https://en.wikipedia.org/wiki/Exponentially_modified_Gaussian_distribution
        
    Returns
    -------
    np.ndarray
        Exponentially modified Gaussian PDF for given params.
    """
    I = lam/2*np.exp(lam/2*(2*mu + lam*sigma**2 - 2*x))
    II = erfc((mu + lam*sigma**2 - x)/(2**0.5 * sigma))
    return I * II

def sum_emg(x : np.ndarray, *args) -> np.ndarray:
    """
    Calculates a sum of EMGs
    
    Parameters
    ----------
    x : np.ndarray
        Time in s at every time bin
    args : np.ndarray
        Parameters of all the EMGs to be summed
        ordered as: a1, mu1, sigma1, a2, mu2, sigma2, etc.
        
    Returns
    -------
    emg : np.ndarray
        Sum of EMGs each scaled by an amplitude. 
    """
    if type(args) != list:
        args = np.array(args).flatten()
        args = list(args)
    scat = args.pop(-1)
    args = np.array(args).flatten()
    num_g = int(len(args)/3)
    args = args.reshape((num_g,3))
    a,mu,sigma = args[...,0], args[...,1], args[...,2]
    emg = np.zeros(len(x))
    for i in range(num_g):
        emg += a[i]*exponorm(x,1/scat, mu[i] ,sigma[i])
    if np.isnan(emg).any() or np.isinf(emg).any():
        emg = np.zeros(len(x))
    return emg
    
def smooth_and_find_peaks(x : np.ndarray, y : np.ndarray, frac : float, 
    l : float, f : int , t_res : float = 2.56e-6) -> tuple:
    """
    Find peaks in the pulse profile using the lowess method.
    
    Parameters
    ----------
    x : np.ndarray
        Independent variable (e.g. timestamps in s)
    y : np.ndarray
        1D pulse profile or spectrum
    frac : float
        Lowess smoothing fraction, e.g. see
        https://towardsdatascience.com/lowess-regression-in-python-how-to-discover-clear-patterns-in-your-data-f26e523d7a35
    l : float
        Lower limit for prominence
    f : int
        np.ceil(7/f)+1 is the minimum distance between peaks
    t_res : float, optional
        Resolution time, default is full baseband resolution.
    
    Returns
    -------
    tuple
        The smooth profile, the peaks found with lowess, their prominences and heights.
    """
    smooth = lowess(y,x,frac = frac)
    x,y_smooth = smooth[...,0], smooth[...,1]
      
    # Find peaks 
    peaks = find_peaks(y_smooth, prominence=0, height=0, distance=np.ceil(7/f)+1)
    peaks, prominences, heights = peaks[0], peaks[1]['prominences'], peaks[1]['peak_heights']
    # Prominence and height kwargs added so that find_peaks returns the prominences and heights
    # Sort peaks by prominence
    pos = np.flip(np.argsort(prominences))
    prominences = np.array([np.ceil(prominences[i]*10)/10 for i in pos])
    peaks = np.array([peaks[i] for i in pos])
    heights = np.array([heights[i] for i in pos])
    print(peaks)
    print(peaks*t_res)
    print(prominences)
    print(heights)
    keep = prominences >= l
    keep = np.logical_and(keep, np.logical_or(heights >= 3, np.logical_and(heights >= 2, prominences >= 1.5)))
    peaks = peaks[keep]
    prominences = prominences[keep]
    heights = heights[keep]
    pos = np.flip(np.argsort(prominences))
    prominences = np.array([prominences[i] for i in pos])
    peaks = np.array([peaks[i] for i in pos])
    heights = np.array([heights[i] for i in pos])
    return y_smooth, peaks, prominences, heights
    
def count_components(profile : np.ndarray, t_res : float = 2.56e-6, ds : int = 1, 
    event_id : str = '', diagnostic_plots : bool = False) -> tuple:
    """
    Calculate various parameters needed for lowess smoothing, then perform the smoothing
    and find the number of burst components to fit for.
    
    Parameters
    ----------
    profile : np.ndarray
        1D pulse profile (where to look for peaks)
    t_res : float, optional
        Time resolution of each time bin in s
    ds : int, optional
        downsampling factor
    event_id : str, optional
        Event ID of the FRB
    diagnostic_plots : bool, optional
        If True, displays plots of intermediate steps.
        
    Returns
    -------
    tuple
       Array of peak times found with lowess, array of peaks excluded based on certain criteria (e.g. faint ones).
    
    """
    x = np.arange(len(profile))
    y = profile
    f = max(1, np.log10(1/(t_res*len(x))))
    if ds > 100:
        f *= np.log10(ds**2)
    if max(profile) > 50:
        l = 0.8
    else:
        l = 0.3
    ub, lb = profile.argmax(), profile.argmax()
    while lb >= 0 and profile[lb] >= max(profile)/2.:
        lb -= 1
    while ub < len(profile) - 1 and profile[ub] >= max(profile)/2.:
        ub += 1
    w = len(profile[lb:ub])
    if max(profile) > 50:
        best_frac = 0.020
        flag = False
    elif max(profile) > 9 and ds <= 64:
        best_frac = 0.015 * f
        flag = True
    elif max(profile) > 6 and ds <= 64:
        if ds <= 20:
            best_frac = 0.015 * f
            flag = True
        else:
            if max(profile) < 9:
                best_frac = 0.035 * f
                flag = False
            else:
                best_frac = 0.015 * f
                flag = True
    else:
        # If faint and wide, just use the pulse FWHM
        best_frac = len(profile[profile >= max(profile)/2.]) * 0.5 / len(x)
        flag = False
    if best_frac * len(x) <= 5:
        if ds <= 64:
            best_frac = 6 / len(x)
            flag = True
        else:
            best_frac *= 2
            flag = False
    y_smooth, peaks, prominences, heights = smooth_and_find_peaks(x,y,best_frac,l,f,t_res=t_res)
    # For faint wide bursts:
    if len(y_smooth[y_smooth >= max(y_smooth)/2.]) / len(x) >= 0.3:
        best_frac = len(y_smooth[y_smooth >= max(y_smooth)/2.]) / len(x)
    y_smooth, peaks, prominences, heights = smooth_and_find_peaks(x,y,best_frac,l,f,t_res=t_res)
    if len(peaks) > 6 and flag and len(x) * best_frac < 10 or len(peaks) > 6 and len(x) * t_res < 0.1:
        while len(peaks) > 6:
            best_frac *= 1.3
            y_smooth, peaks, prominences, heights = smooth_and_find_peaks(x,y,best_frac,l,f,t_res=t_res)
    print('FRAC:', best_frac)
    print(peaks)
    print(peaks*t_res)
    print(prominences)
    print(heights)
    faint = sum(np.logical_or(heights < 4, prominences < 2))
    if diagnostic_plots:
        plt.figure(figsize = (20,10))
        plt.plot(np.arange(len(profile))*t_res, profile, color = 'k', alpha = 0.3, label = 'Event '+event_id)
        plt.plot(x*t_res, y_smooth, color = 'k', label = 'Lowess')
        plt.legend(fontsize = 20)
        for p in peaks:
            plt.axvline(p*t_res, color = 'r', ls='--')
        plt.xticks(fontsize = 17)
        plt.yticks(fontsize = 17)
        plt.xlabel('Time (s)', fontsize = 20)
        plt.show()
        if max(profile) > 100:
            plt.figure(figsize = (20,10))
            plt.plot(np.arange(len(profile))*t_res, profile, color = 'k', alpha = 0.3, label = 'Event '+event_id)
            plt.plot(x*t_res, y_smooth, color = 'k', label = 'Lowess')
            plt.legend(fontsize = 20)
            for p in peaks:
                plt.axvline(p*t_res, color = 'r', ls='--')
            plt.xticks(fontsize = 17)
            plt.yticks(fontsize = 17)
            plt.xlabel('Time (s)', fontsize = 20)
            plt.ylim(-5,20)
            plt.show()
    
    return peaks