# import general packages.
import numpy as np

# import and configure matplotlig for GUI-less node.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import matplotlib.gridspec as gridspec

from fitburst.routines.profile import *
import statsmodels.api as sm
from fitburst.routines.spectrum import rpl
#Can easily replace get_floor by something else
from baseband_analysis.utilities import get_floor
def reduced_chisq(data : np.ndarray, fit : np.ndarray, pars: int) -> float:
    """
    Calculates the reduced chi squared
    
    Parameters
    ----------
    data : np.ndarray
        1D array (e.g. pulse profile or spectrum)
    fit : np.ndarray
        Fit corresponding to the data
    pars : int
        Number of parameters used in the fit model
    Returns
    -------
    float
        Reduced chi squared
    
    """
    m = get_floor(data)
    return sum((data[m]-fit[m])**2)/(len(data[m])-pars)

def show_fit(profile : np.ndarray, xvals : np.ndarray, final_pars : np.ndarray, 
    res : float = 1., event_id : str = '', m : str = 'emg') -> None:
    """
    Display plot with fit against data.
    
    Parameters
    ----------
    profile : np.ndarray
        Data (pulse profile or pulse spectrum)
    x_vals : np.ndarray
        Timestamp of every time bin in s or freq in MHz (same size as profile)
    final_pars : np.ndarray
        Array with the parameters of the fit to the profile
    event_id : str, optional
        Event ID of the FRB
    m : str, optional
        model that is fit, either 'emg' or 'rpl'
    Returns
    -------
    None
        Displays a plot with the data and fit result.   
    """
    if m == 'emg':
        peaks = np.array(final_pars[:-1]).reshape(int((len(final_pars) - 1)/3.),3)[:,1]
        fit = sum_emg(xvals,final_pars)
        npars = len(peaks)*3+1
        mean = np.array(final_pars[:-1]).reshape((len(peaks),3))[...,1]
        print('MEAN', mean)
    else:
        fit = rpl(xvals,final_pars)
        npars = 3
    chisqr = reduced_chisq(profile,fit,npars)
    if m == 'rpl':
        start = min(xvals)
        f = 1
        xlabel = 'Frequency (MHz)'
        d = max(xvals) - min(xvals)
    else:
        start = 0
        f = 1000
        xlabel = 'Time (ms)'
        d = max(xvals) * f
    profile2, t_res2, factor = profile, res, 1#cut_profile(path, downsample = 12)
    fig, ax = plt.subplots(nrows = 2, ncols = 1, sharex = True, figsize = (12,10),gridspec_kw = {'hspace':0,'height_ratios':[3, 1]})
    ax[0].plot(xvals * f,profile2, color = 'k', label = 'Event '+event_id)

    ax[0].plot(f * xvals,np.sqrt(factor)*fit, lw = 2, color = 'r', label = 'Fit')
    if m == 'emg':
        for i in range(len(mean)):
            ax[0].axvline(f * mean[i], color = 'k', alpha = 0.3, ls = '-', lw = 2)
            ax[1].axvline(f * mean[i], color = 'k',alpha = 0.3, ls = '-', lw = 2)
    print('Reduced Chi Squared for Event ' + event_id +' is '+ str(round(chisqr,3)))
    #residuals:
    residuals = profile2-np.sqrt(factor)*fit
    #residuals = residuals[400:(residuals.size // factor) * factor].reshape(-1, factor).mean(axis=1)

    ax[1].scatter(xvals * f, residuals, color = 'k', label = 'Residuals')
    #ax[0].legend(fontsize = 20)
    #ax[1].legend(fontsize = 14)
    plt.axhline(0,color = 'r', lw = 2)
    
    
    ax[1].set_xlabel(xlabel, fontsize = 20)
    ax[0].set_ylabel('S/N', fontsize = 20)
    ax[1].set_ylabel('S/N', fontsize = 20)
    if d > 100:
        ticks = np.arange(start, max(xvals) * f, 50, dtype = int)
    elif d < 100 and d > 50:
        ticks = np.arange(start, max(xvals) * f, 20, dtype = int)
    elif d < 50 and d > 7:
        ticks = np.arange(start, max(xvals) * f, 2, dtype = int)
    elif d < 7 and d > 2:
        ticks = np.arange(start, max(xvals) * f, 1, dtype = int)
    else:
        ticks = np.arange(start, max(xvals) * f, 0.5)
    plt.xticks(ticks,fontsize = 17)
    
    d = max(profile2) - min(profile2)
    if d > 100:
        ticks = np.arange(min(profile2), max(profile2), 50, dtype = int)
    elif d < 100 and d > 50:
        ticks = np.arange(min(profile2), max(profile2), 20, dtype = int)
    elif d < 50 and d > 10:
        ticks = np.arange(min(profile2), max(profile2), 5, dtype = int)
    elif d < 10 and d > 2:
        ticks = np.arange(min(profile2), max(profile2), 1, dtype = int)
    else:
        ticks = np.arange(min(profile2), max(profile2), 0.5)
    ax[0].set_yticks(ticks)
    ax[0].set_yticklabels(ticks, fontsize=17)

    d = max(residuals) - min(residuals)
    if d > 100:
        ticks = np.arange(min(residuals), max(residuals), 50, dtype = int)
    elif d < 100 and d > 50:
        ticks = np.arange(min(residuals), max(residuals), 20, dtype = int)
    elif d < 50 and d > 20:
        ticks = np.arange(min(residuals), max(residuals), 5, dtype = int)
    elif d < 20 and d > 7:
        ticks = np.arange(min(residuals), max(residuals), 2, dtype = int)
    elif d < 7 and d > 2:
        ticks = np.arange(min(residuals), max(residuals), 1, dtype = int)
    else:
        ticks = np.arange(min(residuals), max(residuals), 0.5)
    ax[1].set_yticks(ticks)
    ax[1].set_yticklabels(ticks, fontsize=17)

    plt.yticks(fontsize=17)
    plt.show()
    
    #QQ plot
    #create Q-Q plot with 45-degree line added to plot
    fig = sm.qqplot(residuals, line='45')
    plt.xticks(fontsize = 13)
    plt.yticks(fontsize = 13)
    plt.ylabel('Sample Quantiles',fontsize = 13)
    plt.xlabel('Theoretical Quantiles', fontsize = 13)
    plt.show()

    return

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
    ax[0][0].plot(x,get_profile(power), color = 'k')
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

def plot_chain(param : np.ndarray, param_name : str = 'Parameter') -> None:
    """
    Plot the trace and posterior of a parameter.
    
    Parameters
    ----------
    param : np.ndarray
        Parameter posterior values
    param_name : str, optional
        Name of parameter
    
    Returns
    -------
    None
        Displays plot of trace and posterior for every parameter.
    """
  
    # Summary statistics
    mean = np.mean(param)
    median = np.median(param)
    cred_min, cred_max = np.percentile(param, 16), np.percentile(param, 84)
  
    # Plotting
    plt.figure(figsize = (12,10))
    plt.subplot(2,1,1)
    plt.plot(param, color = 'k')
    plt.xlabel('Samples', fontsize = 18)
    plt.ylabel(param_name, fontsize = 18)
    plt.axhline(mean, color='r', lw=2, linestyle='--')
    plt.axhline(median, color='c', lw=2, linestyle='--')
    plt.axhline(cred_min, linestyle=':', color='k', alpha=0.2)
    plt.axhline(cred_max, linestyle=':', color='k', alpha=0.2)
    plt.title(r'Trace and Posterior Distribution for '+ param_name, fontsize = 18)

    plt.subplot(2,1,2)
    plt.hist(param, 30, density=True,color = 'k'); sns.kdeplot(param, shade=True, color='b')
    plt.xlabel(param_name, fontsize = 18)
    plt.ylabel('Density', fontsize = 18)
    plt.axvline(mean, color='r', lw=2, linestyle='--',label='Mean')
    plt.axvline(median, color='purple', lw=2, linestyle='--',label='Median')
    plt.axvline(cred_min, linestyle=':', color='k', alpha=0.3, label='68% CI')
    plt.axvline(cred_max, linestyle=':', color='k', alpha=0.3)
  
    plt.gcf().tight_layout()
    plt.legend()
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
    norm1 = np.meshgrid(profile_fit/get_profile(model), np.ones(power.shape[0]))[0]
    model *= norm1
    norm2 = np.meshgrid(spectrum_fit/get_spectrum(model), np.ones(power.shape[1]))[0].T
    model *= norm2
    if diagnostic_plots:
        plot_waterfall(power, res, freq, fit_spect = spectrum_fit, fit_profile = profile_fit)
        plt.show()
        plot_waterfall(model, res, freq)
        plt.show()
        plot_waterfall(power, res, freq, fit_spect = get_spectrum(model), fit_profile = get_profile(model))
        plt.show()
        plot_summary_triptych(res, freq, power, model, mask_freq = mask_freq)
    return np.array(mask_freq)
