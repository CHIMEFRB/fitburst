from fitburst.routines.profile import *
from fitburst.routines.spectrum import *
from fitburst.utilities.plotting import *
from fitburst.analysis.mcmc_fitter import *
from fitburst.backend.generic import DataReader
from . import waterfall_plotting


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
    
def fit_profile(path : str, nwalkers : int = 100, nchain : int = 50000, event_id : str = None,
    ncores : int = 10, logl : int = 0, fit_spectrum : bool = False,
    show_chains : bool = False, diagnostic_plots : bool = False) -> tuple:
    """
    Fits an EMG to the pulse profile and then a RPL to the spectrum of every burst component using MCMC.
    If a path is provided in 'save', saves the MCMC fitting results.
   
    Parameters
    ----------
    path : str
        Path to the singlebeam h5 file that contains baseband data.
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
    show_chains : bool, optional
        If true, produce plots of the parameter value vs chain step.
        Default is False.
    diagnostic_plots : bool, optional
        If True, shows plots of many intermediate steps. Default is False.
    Returns
    -------
    Tuple
        All the required data and fit parameters to make the fitburst input file.
    """
    # Fit pulse profile
    if event_id is None:
        event_id = path.split('/')[-1].split('_')[-1].split('.')[0]
    
    # Read .npz file
    # read in data.
    data = DataReader(path)
    # load data into memory and pre-process.
    data.load_data()
    
    # Calculate burst profile out of full data
    profile = get_profile(data.data_full)

    peak_times = data.burst_parameters['arrival_time']

    t_res = data.res_time

    print("Starting MCMC fit")
    profile_pars, profile_pars_errs = fit_emg_mcmc(profile, t_res * np.arange(len(profile)), peaks=peak_times, logl = logl, res = t_res, nwalkers=nwalkers, 
                 nchain=nchain, event_id=event_id, ncores=ncores, show_chains = show_chains, 
                 diagnostic_plots=diagnostic_plots
                 )
    n = len(peak_times)
    tmp_pars = np.reshape(profile_pars[:-1], (n, 3))
    mus, widths = tmp_pars[...,1], tmp_pars[...,2]
    mus = np.append(mus, len(profile)*t_res)
    
    # Fit spectrum for each component
    amps, running, index = np.zeros(n), np.zeros(n), np.zeros(n)
    amps_errs, running_errs, index_errs = np.zeros((n,2)), np.zeros((n,2)), np.zeros((n,2))
    if fit_spectrum:
        profile_fit = sum_emg(t_res * np.arange(power.shape[-1]), profile_pars)
        lims = [np.where(profile_fit > 0.1)[0][0],np.where(profile_fit > 0.1)[0][-1]]
        fit_ps = find_peaks(profile_fit)[0]
        if len(fit_ps) != n:
            fit_ps = np.array(sorted([int(p / t_res) for p in peak_times]))
        print(fit_ps)
        for i in range(n):
            if i == 0:
                tmp = profile_fit[int(fit_ps[i]):int(mus[i+1] / t_res)]
                print(int(fit_ps[i]), int(peak_times[i] / t_res), int(mus[i+1] / t_res))
                all_lims = [lims[0], int(fit_ps[i]) + np.where(tmp == min(tmp))[0][0]]
            elif i < n-1:
                tmp = profile_fit[int(fit_ps[i]):int(mus[i+1] / t_res)]
                if len(tmp) > 2:
                    all_lims.append(int(fit_ps[i]) + np.where(tmp == min(tmp))[0][0])
                else:
                    tmp = profile_fit[int(mus[i]):int(mus[i+1] / t_res)]
                    all_lims.append(int(mus[i]) + np.where(tmp == min(tmp))[0][0])
            else:
                all_lims.append(lims[-1])
        for i in range(n):
            lim = [all_lims[i], all_lims[i+1]]
            plt.clf()
            waterfall_plotting.plot_waterfall(power, t_res, freq, peaks = np.array(lim)*t_res)
            plt.show()

            spectrum = get_spectrum(power[...,max(0,lim[0] - int(widths[i] / t_res)): lim[1]])

            spect_peaks = spect_count_components(spectrum, freq, event_id=event_id, diagnostic_plots=diagnostic_plots)

            spectrum_pars, spectrum_errs = fit_rpl_mcmc(spectrum, freq, peaks=spect_peaks, res = 400. / 1024., nwalkers=100, 
                     nchain=5000, event_id=event_id, ncores=ncores, show_chains = show_chains,
                     diagnostic_plots=diagnostic_plots
                     )
            amps[i] = spectrum_pars[0]
            index[i] = spectrum_pars[1]
            running[i] = spectrum_pars[2]

            amps_errs[i] = spectrum_errs[0]
            index_errs[i] = spectrum_errs[1]
            running_errs[i] = spectrum_errs[2]   

    spectrum_pars = [amps, index, running]
    spectrum_errs = [amps_errs, index_errs, running_errs]
    return data, profile_pars, spectrum_pars
