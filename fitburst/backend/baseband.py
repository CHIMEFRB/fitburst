from baseband_analysis.core import BBData
from baseband_analysis.utilities import get_profile, get_snr, get_floor, get_main_peak_lim, apply_coherent_dedisp, tiedbeam_baseband_to_power
from fitburst.routines.profile import *
from fitburst.routines.spectrum import *
from fitburst.utilities.plotting import *
from fitburst.analysis.mcmc_fitter import *
from . import waterfall_plotting


def cut_profile(path : str, downsample : int = None, downsample2 : int = None, peaks : list = None, 
    time_range : tuple = None, DM : float = None, fit_DM : bool = True, spectrum_lim : bool = True, 
    fill_missing_time : bool = None, return_full : bool = True, diagnostic_plots : bool = False) -> tuple:
    """
    Load and clean the data, correct DM, and find the burst components
   
    Parameters
    ----------
    path : str
        Path to the singlebeam h5 file that contains baseband data.
    downsample : int, optional
        To downsample the data in addition to the downsampling in the singlebeam file.
        This downsampling factor is used in the DM correction and in the peak finding steps.
    downsample2 : int, optional
        To downsample the data by a different factor than 'downsample' for all the steps after peak finding.
        This is useful if you want to downsample more to find peaks more easily, but then downsample less to
        make MCMC fitting easier.
    peaks : list, optional
        List of bin numbers at downsampling 'downsample' where to put the initial guess
        for the peak position of each component. If not provided, use lowess algorithm.
    time_range : tuple, optional
        Bin numbers where to start and end the baseband data at downsampling 'downsample'.
    DM : float, optional
        If DM_phase and/or the DM_correction seem wrong, you can input your own DM.
    fit_DM : bool, optional
        If True, run DM_phase's structure maximising DM correction and apply it.
    spectrum_lim : bool, optional
        Whether to cut out the frequency channels without strong signal (True) or not. 
        Default is True.
    fill_missing_time : bool, optional
        If you want to force get_snr to apply the procedue to fill the triangular artefact with noise.
        Default is None (i.e. get_snr decides whether to fill the triangle or not). 
    return_full : bool, optional
        If True (default), returns data, freq_id, freq, power[...,start:end], valid_channels, 
        DM, DM_err, downsampling_factor, profile, t_res, peak_times, start * t_res, power
        If False, returns only pulse profile and time resolution
    diagnostic_plots : bool, optional
        If True, shows plots of many intermediate steps. Default is False.
    Returns
    -------
    Tuple
        See description for return_full
    
    """
    data = BBData.from_file(path)
    event_id = path.split('/')[-2].split('_')[-1]
    fname = pathlib.Path(path)
    assert fname.exists(), f'No such file: {fname}'  # check that the file exists
    mtime = datetime.datetime.fromtimestamp(fname.stat().st_mtime)
    if mtime >= datetime.datetime(2020, 5, 1):
        if DM is not None:
            try:
                data['tiedbeam_power']
            except KeyError:
                tiedbeam_baseband_to_power(data,time_downsample_factor=1,dm = DM,dedisperse=True)           
            apply_coherent_dedisp(data, DM)
            try:
                data['tiedbeam_power'].attrs['DM_coherent']
            except KeyError:
                print('Coherent de-dispersion failed.')
            else:
                print('Coherent de-dispersion performed successfully.')
        else:
            raise ValueError('Please supply the DM to use for coherent de-dispersion!')
    freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = get_snr(
        data, 
        DM = DM,
        downsample=downsample,
        fill_missing_time = fill_missing_time,
        diagnostic_plots=False,
        spectrum_lim = spectrum_lim,
        return_full=True
        )
    dm_max_range = 0.3

    profile = get_profile(power)
    if time_range is None:
        start, end = get_signal(profile, ds = downsampling_factor)
    else:
        start, end = time_range[0], time_range[1]
    plt.clf()
    if fit_DM:
        DM_corr, DM_err = get_structure_max_DM(power[...,start: end], freq, t_res = 2.56e-6 * downsampling_factor, DM_range = dm_max_range)
        plt.show()
    else:
        DM_corr ,DM_err = 0, 0.05/2
    freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = get_snr(
        data,
        DM = DM + DM_corr,
        downsample=downsample,
        fill_missing_time = fill_missing_time,
        spectrum_lim = spectrum_lim,
        diagnostic_plots=False,
        return_full=True
    )
    t_res = 2.56e-6 * downsampling_factor
    profile = get_profile(power)
    if time_range is None:
        start, end = get_signal(profile, ds = downsampling_factor)
    else:
        start, end = time_range[0], time_range[1]
    profile = profile[start:end]
    if peaks is None:
        peaks = count_components(profile, t_res, ds = downsampling_factor, diagnostic_plots=diagnostic_plots)
    else:
        peaks = np.array(peaks)
    if downsample2 is not None:
        plt.clf()
        print(start, end)
        waterfall_plotting.plot_waterfall(power[...,start:end],t_res, freq)
        plt.show()
        old_ds = downsampling_factor
        freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = get_snr(
        data,
        DM = DM,
        downsample=downsample2,
        fill_missing_time = fill_missing_time,
        spectrum_lim = spectrum_lim,
        diagnostic_plots=False,
        return_full=True
        )
        profile = get_profile(power)
        f = old_ds/downsampling_factor
        start, end = int(start*f), int(min(end*f, len(profile)))
        plt.clf()
        if fit_DM:
            DM_corr, DM_err = get_structure_max_DM(power[...,start: end], freq, t_res = 2.56e-6 * downsampling_factor, DM_range = 0.3)
            plt.show()
        else:
            DM_corr ,DM_err = 0, 0.05/2
        freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = get_snr(
        data,
        DM = DM + DM_corr,
        downsample=downsample2,
        fill_missing_time = fill_missing_time,
        spectrum_lim = spectrum_lim,
        diagnostic_plots=False,
        return_full=True
        )
        profile = get_profile(power)
        profile = profile[start:end]
    
        t_res = 2.56e-6 * downsampling_factor
        peak_times = peaks * f * t_res
    else:
        peak_times = peaks * t_res
        
    if diagnostic_plots:
        plt.clf()
        waterfall_plotting.plot_waterfall(power[...,start:end],t_res, freq)
        plt.show()
        plt.clf()
        waterfall_plotting.plot_waterfall(power[...,start:end],t_res, freq,peaks = peak_times)
        plt.show()
    if return_full:
        return data, freq_id, freq, power[...,start:end], valid_channels, DM, DM_err, downsampling_factor, profile, t_res, peak_times, start * t_res, power
    else:
        return profile, t_res
        
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
    
def fit_profile(path : str, nwalkers : int = 100, nchain : int = 500000, 
    ncores : int = 10, time_range : tuple = None, downsample : int = None, 
    downsample2 : int = None, DM : float = None, fit_DM : bool = True,
    logl : int = 0, save : bool = None, spectrum_lim : bool = True, 
    fill_missing_time : bool = None, show_chains : bool = False, 
    peaks : list = None, diagnostic_plots : bool = False) -> tuple:
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
    time_range : tuple, optional
        Bin numbers where to start and end the baseband data at downsampling 'downsample'.
    downsample : int, optional
        To downsample the data in addition to the downsampling in the singlebeam file.
        This downsampling factor is used in the DM correction and in the peak finding steps.
    downsample2 : int, optional
        To downsample the data by a different factor than 'downsample' for all the steps after peak finding.
        This is useful if you want to downsample more to find peaks more easily, but then downsample less to
        make MCMC fitting easier.
    DM : float, optional
        If DM_phase and/or the DM_correction seem wrong, you can input your own DM.
    fit_DM : bool, optional
        If True, run DM_phase's structure maximising DM correction and apply it.
    logl : int, optional
        Two possible values:
        0 : default wide flat prior for params
        1 : narrower flat prior (useful for when the MCMC has trouble converging)
    save : str, optional
        To save the params from the MCMC fits, provide a path where to save the results.
    spectrum_lim : bool, optional
        Whether to cut out the frequency channels without strong signal (True) or not. 
        Default is True.
    fill_missing_time : bool, optional
        If you want to force get_snr to apply the procedue to fill the triangular artefact with noise.
        Default is None (i.e. get_snr decides whether to fill the triangle or not). 
    show_chains : bool, optional
        If true, produce plots of the parameter value vs chain step.
        Default is False.
    peaks : list, optional
        List of bin numbers at downsampling 'downsample' where to put the initial guess
        for the peak position of each component. If not provided, use lowess algorithm.
    diagnostic_plots : bool, optional
        If True, shows plots of many intermediate steps. Default is False.
    Returns
    -------
    Tuple
        All the required data and fit parameters to make the fitburst input file.
    """
    # Fit pulse profile
    event_id = path.split('/')[-2].split('_')[-1]
    data, freq_id, freq, power, valid_channels, DM, DM_err, downsampling_factor, profile, t_res, peak_times, start, full_power = cut_profile(path, diagnostic_plots = diagnostic_plots, time_range = time_range, fill_missing_time = fill_missing_time, downsample = downsample, downsample2 = downsample2, DM = DM, fit_DM = fit_DM, peaks = peaks, spectrum_lim = spectrum_lim, return_full = True)
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
    # For plotting only
    if n > 1:
        spectrum_pars_plot, _ = fit_rpl_mcmc(get_spectrum(power), freq, peaks=spect_peaks, res = 400. / 1024., nwalkers=100, 
                 nchain=10000, event_id=event_id, ncores=ncores, show_chains = show_chains,
                 diagnostic_plots=diagnostic_plots
                 )
    else:
        spectrum_pars_plot = spectrum_pars
    waterfall_plotting.plot_waterfall(power, t_res, freq, fit_spect = rpl(freq, spectrum_pars_plot) , fit_profile = profile_fit)
    plt.show()
    mask_freq = np.array(np.zeros(power.shape[0]), dtype = bool)
    mask = np.array(np.ones(power.shape[0]), dtype = bool)
    for i in range(power.shape[0]):
        if i in mask_freq:
            mask[i] = False
    power[~mask] = 0
    spectrum_pars = [amps, index, running]
    spectrum_errs = [amps_errs, index_errs, running_errs]
    if save is not None:
        np.savez(save + event_id, {'profile_pars':profile_pars, 'profile_pars_errs': profile_pars_errs, 'spectrum_pars': spectrum_pars, 'spectrum_pars_errs': spectrum_errs, 'DM': DM, 'DM_err': DM_err, 'S/N': max(profile), 'bw': max(freq) - min(freq), 'ds': downsampling_factor})
    return data, freq_id, freq, power, DM, downsampling_factor, t_res, profile_pars, spectrum_pars, mask_freq
