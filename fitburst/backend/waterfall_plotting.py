# import general packages.
import numpy as np

# import and configure matplotlig for GUI-less node.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

from fitburst.utilities.plotting import *
# These two functions can easily be replaced / copied from baseband-analysis
from baseband_analysis.core.signal import _get_profile, get_spectrum

def plot_waterfall(power : np.ndarray, t_res : float, freq : np.ndarray, 
    fit_spect : np.ndarray = None, fit_profile : np.ndarray = None, 
    peaks : np.ndarray = None) -> None:
    """
    Plots a waterfall of the input
    
    Parameters
    ----------
    power : np.ndarray
        2D (nfreq, ntime) power of the data
    t_res : float
        Resolution time in s of time
    freq : np.ndarray
        Array with the value of the frequency in MHz at each freq bin
    fit_spect : np.ndarray, optional
        Array with the fit to the spectrum (same size as freq)
    fit_profile : np.ndarray, optional
        Array with the fit to the profile (size = ntime)
    peaks : np.ndarray, optional
        Array with the peak times of every burst component in s
    
    Returns
    -------
    None
        Displays a waterfall plot
    
    """
    f, ax = plt.subplots(2, 2, figsize = (12,12), gridspec_kw={'width_ratios': [3, 1], 'height_ratios': [1, 3]})
    fs = 20
    extent = [0, power.shape[-1] * t_res * 1000, min(freq), max(freq)]
    mean = np.nanmean(power)
    std = np.nanstd(power)
    ax[1][0].imshow(power, cmap="viridis", origin="upper", aspect="auto", interpolation="nearest", extent = extent, vmin = mean - 3 * std, vmax = mean + 3 * std)
    ax[1][0].set_ylabel(r'Frequency (MHz)', fontsize = fs)
    ax[1][0].set_xlabel(r'Time (ms)', fontsize = fs)

    x = np.arange(power.shape[-1]) * t_res * 1000
    ax[0][0].plot(x,_get_profile(power), color = 'k')
    if fit_profile is not None:
        ax[0][0].plot(x,fit_profile, color = 'r')
    ax[0][0].set_xlim(0, power.shape[-1] * t_res * 1000)
    
    ax[1][1].plot(get_spectrum(power), np.flip(np.arange(power.shape[0])), color = 'k')
    f.subplots_adjust(wspace=0, hspace=0)
    if fit_spect is not None:
        ax[1][1].plot(fit_spect, np.flip(np.arange(power.shape[0])), color = 'r')
    ax[1][1].set_ylim(0, power.shape[0])
    
    if peaks is not None:
        for p in peaks:
            ax[0][0].axvline(p * 1000, ls = '--' , c = 'r')
            ax[1][0].axvline(p * 1000, ls = '--' , c = 'r')

    plt.setp(ax[1][0].get_xticklabels(), fontsize=fs-2)
    plt.setp(ax[0][0].get_xticklabels(), fontsize=fs-2)
    plt.setp(ax[1][1].get_xticklabels(), fontsize=fs-2)

    plt.setp(ax[1][0].get_yticklabels(), fontsize=fs-2)
    plt.setp(ax[0][0].get_yticklabels(), fontsize=fs-2)
    plt.setp([ax[1][1].get_yticklabels()], fontsize=fs-2)

    ax[0][0].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax[1][1].tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are on
        labelleft=False) # labels along the bottom edge are off

    #Remove plot in upper right corner
    ax[0][1].tick_params(
        axis='x',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        bottom=False,      # ticks along the bottom edge are off
        top=False,         # ticks along the top edge are off
        labelbottom=False) # labels along the bottom edge are off

    ax[0][1].tick_params(
        axis='y',          # changes apply to the x-axis
        which='both',      # both major and minor ticks are affected
        left=False,      # ticks along the bottom edge are off
        right=False,         # ticks along the top edge are on
        labelleft=False) # labels along the bottom edge are off
    ax[1][1].xaxis.set_major_locator(MaxNLocator(nbins=3,prune='lower'))
    for spine, i in zip(ax[0][1].spines.values(), [0,1,2,3]):
        if i != 0:
            spine.set_edgecolor('white')

    #plt.savefig('RA-Dec.png', dpi = 400)
    plt.show()
    return

def make_waterfall(res : int, profile_pars : np.ndarray, freq : np.ndarray, 
    spectrum_pars : np.ndarray, power : np.ndarray, 
    diagnostic_plots : bool = True) -> np.ndarray:
    """
    Make a 2D model based on 1D profile and 1D spectrum (just for fun).
    
    Parameters
    ----------
    res : int
        Time resolution in s
    profile_pars : np.ndarray
        Parameters of EMG MCMC fit
    freq : np.ndarray
        Array of frequency bin values in MHz      
    spectrum_pars : np.ndarray
        Parameters of RPL spectrum fit
    power : np.ndarray
        2D (nfreq, ntime) power of data
    diagnostic_plots : bool, optional
        If True, show more plots
        
    Returns
    -------
    np.ndarray
        Array of masked freqs
    
    """
    profile_fit = sum_emg(res * np.arange(power.shape[-1]), profile_pars)
    spectrum_fit = np.zeros(power.shape[0])
    for i in range(len(spectrum_pars[0])):
        spectrum_fit += rpl(freq, [spectrum_pars[0][i],spectrum_pars[1][i], spectrum_pars[2][i]])
    spectrum = np.flip(get_spectrum(power))
    mask_freq = []
    std = np.std(abs(spectrum_fit-spectrum))
    print(std)
    for i in range(len(spectrum_fit)):
        if False:#spectrum[i] < spectrum_fit[i] - 3*std:
            mask_freq.append(int(len(spectrum) - 1 - i))
    x, y = np.meshgrid(profile_fit, spectrum_fit)
    model = np.multiply(x, y)
    plt.clf()
    plt.imshow(model)
    plt.show()
    #Rescale model
    norm1 = np.meshgrid(profile_fit/_get_profile(model), np.ones(power.shape[0]))[0]
    model *= norm1
    norm2 = np.meshgrid(spectrum_fit/get_spectrum(model), np.ones(power.shape[1]))[0].T
    model *= norm2
    if diagnostic_plots:
        plot_waterfall(power, res, freq, fit_spect = spectrum_fit, fit_profile = profile_fit)
        plt.show()
        plot_waterfall(model, res, freq)
        plt.show()
        plot_waterfall(power, res, freq, fit_spect = get_spectrum(model), fit_profile = _get_profile(model))
        plt.show()
        plot_summary_triptych(res, freq, power, model, mask_freq = mask_freq)
    return np.array(mask_freq)
