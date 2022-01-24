import numpy as np
import emcee
import seaborn as sns
from multiprocessing import Pool
import matplotlib.pyplot as plt
import statsmodels.api as sm
import corner
from fitburst.routines.profile import *
from fitburst.analysis.fitter import *
from fitburst.routines.spectrum import *
from . import fitter_plotting
from . import fitter

from scipy.signal import find_peaks

def mcmc(popt : np.ndarray, profile : np.ndarray, xvals : np.ndarray, res : float, 
    m: str = 'emg', nwalkers : int = 100, nchain : int = 500000, ncores : int = 10, 
    logl : int = 0, event_id : str = '', m_peaks : np.ndarray = None, 
    plot : bool = True, show_chains : bool = False) -> tuple:
    
    """
    Performs MCMC sampling for a given model and data
    
    Parameters
    ----------
    popt : np.ndarray
        LS best fit parameters, used as initial condition for the MCMC fit.
    profile : np.ndarray
        The 1D data to be fit (either pulse profile or spectrum)
    xvals : np.ndarray
        Timestamp of every time bin if for EMG,
        frequency of every freq bin if for RPL (same size as profile)
    res : float
        Resolution (time for EMG, frequency for RPL)
    m : str
        Which model to fit.
    nwalkers : int, optional
        Number of walkers in MCMC process, default is 50
    nchain : int, optional
        Length of chain in MCMC process, default is 500k
    ncores : int, optional
        Number of frb-analysis cores that are used for the MCMC fitting, 
        default is 10, in general, do not use more than ~15.
    logl : int, optional
        Two possible values:
        0 : default wide flat prior for params
        1 : narrower flat prior (useful for when the MCMC has trouble converging)
    event_id : str, optional
        Event ID of the FRB
    m_peaks : np.ndarray, optional
        Array of peak locations, required if logl = 1 only.
        Used as mu upper bounds in prior.
    show_chains : bool, optional
        If true, produce plots of the parameter value vs chain step.
        Default is False.
    plot : bool, optional
        If True, shows corner plot of the fit. Default is True.
    Returns
    -------
    Tuple
    """
    #MCMC Implementation
    if ncores > 15:
        raise ValueError("Too many cores! Reduce the number of cores to be used to be < 15.")
    global model
    if m == 'emg':
        model = sum_emg
        if m_peaks is not None:
            global max_peaks
            max_peaks = m_peaks
        peaks = np.array(popt[:-1]).reshape(int((len(popt) - 1)/3.),3)[:,1]
        if logl == 0:
            log_likelihood = log_likelihood_emg
        elif logl == 1:
            log_likelihood = log_likelihood_emg_peak_lim
        else:
            log_likelihood = log_likelihood_emg
    elif m == 'rpl':
        model = rpl
        log_likelihood = log_likelihood_rpl
    else:
        model = sum_emg
        log_likelihood = log_likelihood_emg
        print('There is no such model ' + m + '. Using EMG by default.')
    global x
    x = xvals
    global prof
    prof = profile
    
    
    ndim, nwalkers = len(popt), nwalkers
    initial_walkers = np.random.normal(scale=1e-10, loc = popt, size=(nwalkers, ndim))
    with Pool(processes=ncores) as pool:
        sampler = emcee.EnsembleSampler(nwalkers, ndim, log_likelihood, pool = pool)
        sampler.run_mcmc(initial_walkers, nchain, progress=True)        
    try:
        tau = sampler.get_autocorr_time()
        dis = int(2*np.mean(tau))
        thin = int(np.mean(tau)/2)
        print(tau)
    except Exception as e:
        print(e)
        dis = 1000
        thin = 250
        print('Discarded ' + str(dis) + ' and thinned by ' + str(thin))
        tau = np.nan
    flat_samples = sampler.get_chain(discard=dis, thin=thin, flat=True)
    final_pars = []
    final_errs = []
    for i in range(len(popt)):
        mcmc = np.percentile(flat_samples[:, i], [16, 50, 84])
        q = np.diff(mcmc)
        final_pars.append(mcmc[1])
        final_errs.append(np.array(q))
    if plot:
        labels = []
        if m == 'emg':
            ls = ['$A_{', '$\mu_{', '$\sigma_{']
            b = '}$'
            for i in range(len(peaks)):
                labels.extend([l + str(i) + b for l in ls])
                if i == len(peaks)-1:
                    labels.extend(['t'])
            fname = str(len(peaks))
        else:
            labels = ['A', '$\gamma$', 'r']
            fname = 'spectrum'

        fig = corner.corner(flat_samples, labels=labels, title_fmt=".5f", truths=final_pars, truth_color = 'r', show_titles=True, quantiles=[0.16, 0.5, 0.84], title_kwargs={"fontsize": 18})
        plt.savefig('corner'+fname+'.png')
        plt.show()

    fitter_plotting.show_fit(profile, xvals, final_pars, res, event_id, m = m)
    if show_chains:
        for i in range(len(popt)):
            fitter_plotting.plot_chain(flat_samples[:, i],labels[i])
    return tau, flat_samples, final_pars, final_errs
    
def log_likelihood_rpl(pars : np.ndarray) -> float:
    """
    Calculate MSE log likelihood for proposed model parameters with flat wide prior
    
    Parameters
    ----------
    pars : np.ndarray
        Array of parameters for rpl model (see description for rpl)
    
    Returns
    -------
    ret : float
        - np.inf if the proposed parameters are outside of prior
        - sum of squared residuals otherwise.
    """
    ret = - np.sum((prof - model(x,pars))**2)
    # prof and model are global
    if np.isnan(ret).all() or (abs(pars) > 1e4).any():
        ret = - np.inf
    return ret        
        
def log_likelihood_emg(parameters : np.ndarray) -> float:
    """
    Calculate MSE log likelihood for proposed model parameters with flat wide prior
    
    Parameters
    ----------
    parameters : np.ndarray
        Array of parameters for emg model (see description for emg)
    
    Returns
    -------
    ret : float
        - np.inf if the proposed parameters are outside of prior
        - sum of squared residuals otherwise.
    
    """
    pars = np.array(parameters[:-1])
    n = int(len(pars)/3)
    pars = pars.reshape((n ,3))
    if (pars[...,1] < 0).any() or (pars[...,1] > max(x)).any():
        ret = - np.inf
    else:
        ret = - np.sum((prof - model(x,parameters))**2)
        if np.isnan(ret).all():
            ret = - np.inf
    return ret

def log_likelihood_emg_peak_lim(parameters):
    """
    Calculate MSE log likelihood for proposed model parameters with tighter prior
    Use this if there are burst components very nearby and the MCMC has trouble 
    converging on the correct ones, e.g. if there are multiple islands in the corner plot.
    To be used with care!!!
    Parameters
    ----------
    parameters : np.ndarray
        Array of parameters for emg model (see description for emg)
    
    Returns
    -------
    ret : float
        - np.inf if the proposed parameters are outside of prior
        More specifically, unlike the other emg likelihood, this one puts mu upper bound to be the burst peak
        (Gaussian mu is always before the peak location)
        - sum of squared residuals otherwise.
    """
    pars = np.array(parameters[:-1])
    n = int(len(pars)/3)
    pars = pars.reshape((n ,3))
    if (pars[...,1] < 0).any() or (pars[...,1] > max(x)).any() or  (pars[...,1] > max_peaks).any():
        ret = - np.inf
    else:
        ret = - np.sum((prof - model(x,parameters))**2)
        if np.isnan(ret).all():
            ret = - np.inf
    return ret

def fit_emg_mcmc(profile : np.ndarray, xvals : np.ndarray, peaks : np.ndarray, 
    nwalkers : int = 100, nchain : int = 500000, ncores : int = 10, res : float = 1.,
    logl : int = 0, ICs : np.ndarray = None, event_id : str = '',show_chains : bool = False,
    diagnostic_plots : bool = False) -> tuple:
    """
    Fit an EMG model to the profile using MCMC
    
    Parameters
    ----------
    profile : np.ndarray
        Pulse profile array
    xvals : np.ndarray
        Timestamps for every time bin (same length as profile)
    peaks : np.ndarray
        Array with timestamps of the peaks of every burst component
    nwalkers : int, optional
        Number of walkers in MCMC process, default is 50
    nchain : int, optional
        Length of chain in MCMC process, default is 500k
    ncores : int, optional
        Number of frb-analysis cores that are used for the MCMC fitting, 
        default is 10, in general, do not use more than ~15.
    res : float, optional
        Resolution (either time in s or freq in MHz)
    logl : int, optional
        Two possible values:
        0 : default wide flat prior for params
        1 : narrower flat prior (useful for when the MCMC has trouble converging)
    ICs : np.ndarray, optional
        You can provide an array with ICs if the ones calculated by the code are not good enough.
        Provide parameters for each burst component in order, for EMG, the order is: ["A", "mu", "sigma", "lam"]
        Note: This feature doesn't work currently.
    event_id : str, optional
        Event ID of the FRB
    show_chains : bool, optional
        If true, produce plots of the parameter value vs chain step.
        Default is False.
    diagnostic_plots : bool, optional
        If True, shows plots of many intermediate steps. Default is False.
        
    Returns
    -------
    Tuple
        np.ndarray of parameters and np.ndarray of corresponding 1 sigma errors.
    
    """
    # If there are a lot of peaks, fit a range, if there is a 'reasonable' amount of peaks, fit all of them to save time.
    if len(peaks) > 4:
        # This is the index of the last peak we include in the fit 
        # i.e. the last element of num_fit = fit with all components.
        num_peaks_to_fit = np.arange(max(int(len(peaks)/2),4), len(peaks) + 1) 
        log.info("Perform MCMC fit for " + str(num_peaks_to_fit[0]) + " to " + str(num_peaks_to_fit[-1]) + " components.")
    else:
        num_peaks_to_fit = np.array([len(peaks)])
    # Record reduced chi squared as a function number of peaks:
    chi_sqr = np.zeros(len(num_peaks_to_fit))
    all_mcmc_pars = []
    all_mcmc_pars_errs = []
    for i in range(len(num_peaks_to_fit)):
        input_bounds = {}
        peaks_to_fit = peaks[:num_peaks_to_fit[i]]
        print('3 ', input_bounds)
        print(peaks_to_fit)
        fit, popt, popt_e,pcov = fitter.fit_LS(profile, xvals, peaks_to_fit, event_id, ICs = ICs, model='emg', res=res)
        check = find_peaks(fit)[0]
        widths = np.reshape(popt[:-1],(len(peaks_to_fit),3))[...,2]
        it = 1
        while len(check) != len(peaks_to_fit) and it < 5:
            fit, popt, popt_e,pcov = fitter.fit_LS(profile, xvals, peaks_to_fit, event_id, ICs = ICs, tight = it * min(widths), model='emg', res=res)
            check = find_peaks(fit)[0]
            it += 1
        tau, flat_samples, final_pars, final_errs = mcmc(popt, 
            profile, xvals, res,event_id = event_id, m_peaks = peaks_to_fit, logl = logl, nwalkers=nwalkers, nchain=nchain, 
            ncores=ncores, plot=True, show_chains=show_chains)
        all_mcmc_pars.append(final_pars)
        all_mcmc_pars_errs.append(final_errs)
        mcmc_fit = sum_emg(xvals,final_pars)
        chi_sqr[i] = fitter_plotting.reduced_chisq(profile,mcmc_fit,len(peaks_to_fit)*3+1)
    if len(num_peaks_to_fit) > 1:            
        #Determine best number of components by taking the derivative
        derivative = abs(np.diff(chi_sqr))
        if derivative[-1] < 1 and len(num_peaks_to_fit) > 3:
            best_number_of_peaks = num_peaks_to_fit[min(np.where(derivative == max(derivative))[0][0] + 2, len(num_peaks_to_fit)-1)] + 1
            if best_number_of_peaks > max(num_peaks_to_fit):
                best_number_of_peaks = num_peaks_to_fit[np.where(chi_sqr == min(chi_sqr))[0][0]]
        else:
            best_number_of_peaks = num_peaks_to_fit[np.where(chi_sqr == min(chi_sqr))[0][0]]
        print(best_number_of_peaks)
        if diagnostic_plots:
            plt.figure(figsize=(12,10))
            plt.scatter(num_peaks_to_fit, chi_sqr)
            plt.axvline(best_number_of_peaks, color = 'r', ls = '--')
            plt.xlabel('Number of components', fontsize = 20)
            plt.ylabel('Reduced Chi Squared', fontsize = 20)
            plt.show()
        return_pars = np.array(all_mcmc_pars[min(np.where(derivative == max(derivative))[0][0] + 2, len(num_peaks_to_fit)-1)]).flatten()
        return_errs = np.array(all_mcmc_pars_errs[min(np.where(derivative == max(derivative))[0][0] + 2, len(num_peaks_to_fit)-1)]).flatten()
    else:
        return_pars = np.array(all_mcmc_pars).flatten()
        return_errs = np.array(all_mcmc_pars_errs).flatten()
    return return_pars, return_errs

def fit_rpl_mcmc(profile : np.ndarray, xvals : np.ndarray, peaks : np.ndarray, 
    res : float = 400./1024., nwalkers : int = 100, nchain : int = 500000, ncores : int = 10, 
    ICs : np.ndarray = None, event_id : str = '', show_chains : bool = False, 
    diagnostic_plots : bool = False) -> tuple:
    """
    Load and clean the data, correct DM, and find the burst components
   
    Parameters
    ----------
    profile : np.ndarray
        Pulse profile array
    xvals : np.ndarray
        Timestamps for every time bin (same length as profile)
    peaks : np.ndarray
        Array with timestamps of the peaks of every burst component
    res : float, optional
        Frequency resolution in MHz
    nwalkers : int, optional
        Number of walkers in MCMC process, default is 50
    nchain : int, optional
        Length of chain in MCMC process, default is 500k
    ncores : int, optional
        Number of frb-analysis cores that are used for the MCMC fitting, 
        default is 10, in general, do not use more than ~15.     
    ICs : np.ndarray, optional
        You can provide an array with ICs if the ones calculated by the code are not good enough.
        Provide parameters for each burst component in order, for RPL, the order is: [A, spectral index, spectral running]
        Note: This feature doesn't work currently. 
    event_id : str, optional
        Event ID of the FRB
    show_chains : bool, optional
        If true, produce plots of the parameter value vs chain step.
        Default is False.
    diagnostic_plots : bool, optional
        If True, shows plots of many intermediate steps. Default is False.
        
    Returns
    -------
    Tuple
        np.ndarray of parameters and np.ndarray of corresponding 1 sigma errors.
    """
    fit, popt, popt_e,pcov = fitter.fit_LS(profile, xvals, peaks, event_id, ICs = ICs, model='rpl', res=res)
    tau, flat_samples, final_pars, final_errs = mcmc(popt, 
        profile, xvals, res,event_id = event_id, nwalkers=nwalkers, nchain=nchain, 
        ncores=ncores, m = 'rpl', plot=True, show_chains=show_chains)
    mcmc_fit = rpl(xvals,final_pars)
    return final_pars, final_errs