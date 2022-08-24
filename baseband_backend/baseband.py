import numpy as np
import matplotlib.pyplot as plt
import pathlib
from baseband_analysis.core import BBData
#from baseband_analysis.utilities import get_profile, get_snr, get_floor, get_main_peak_lim, coherent_dedisp, tiedbeam_baseband_to_power
from baseband_analysis.core.signal import get_profile, get_floor, get_main_peak_lim, tiedbeam_baseband_to_power
from baseband_analysis.analysis.snr import get_snr
from baseband_analysis.core.dedispersion import coherent_dedisp
from . import waterfall_plotting
from DM_phase import get_dm
from fitburst.routines.profile import *

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


def read_baseband(path : str, downsample : int = None, downsample2 : int = None, peaks : list = None, 
    time_range : tuple = None, freq_range : tuple = None, DM : float = None, fit_DM : bool = True, spectrum_lim : bool = True, 
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
    if DM is not None:
        dm_range_snr = None
        print("Since the DM was already provided get_snr() will run with DM Range as", dm_range_snr)
    else:
        dm_range_snr = 5
    print("While running the first get_snr it will use the DM range as", dm_range_snr)
    try:
        data['tiedbeam_power'].attrs['DM_coherent']
    except KeyError:
        print('Coherent de-dispersion was not done on this event.')
        if DM is not None:
            print('Performing coherent de-dispersion...')
            tiedbeam_baseband_to_power(data,time_downsample_factor=1,dm = DM,dedisperse=True)           
            coherent_dedisp(data, DM)
            try:
                data['tiedbeam_power'].attrs['DM_coherent']
            except KeyError:
                print('Coherent de-dispersion failed.')
            else:
                print('Coherent de-dispersion performed successfully.')
        else:
            raise ValueError('Please supply the DM to use for coherent de-dispersion!')
    print("Running the first get_snr() at DM ", DM)
    freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = get_snr(
        data, 
        DM = DM,
        downsample=downsample,
        fill_missing_time = fill_missing_time,
        diagnostic_plots=False,
        spectrum_lim = spectrum_lim,
        return_full=True,
        DM_range = dm_range_snr,
        lte_mask  = True,
        raise_missing_signal = False,
        do_rfi_clean = True
        )
    dm_max_range = 0.3
    dm_range_snr = None
    print("From now on the DM_range for get_snr() will be", dm_range_snr)

    profile = get_profile(power)
    if time_range is None:
        start, end = get_signal(profile, ds = downsampling_factor)
    else:
        start, end = time_range[0], time_range[1]

    if fit_DM:
        DM_corr, DM_err = get_structure_max_DM(power[...,start: end], freq, t_res = 2.56e-6 * downsampling_factor, DM_range = dm_max_range)
        print("DM after struc max", DM + DM_corr)
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
        return_full=True,
        DM_range = dm_range_snr,
        lte_mask  = True,
        raise_missing_signal = False,
        do_rfi_clean = True
    )
    DM = DM + DM_corr #Changing DM value for fit incase of downsample2
    print("DM value used from now on is", DM)
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
        print("Performing Get snr for downsample2 with DM {} and DM range {}".format(DM, dm_range_snr))
        freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = get_snr(
        data,
        DM = DM,
        downsample=downsample2,
        fill_missing_time = fill_missing_time,
        spectrum_lim = spectrum_lim,
        diagnostic_plots=False,
        return_full=True,
        DM_range = dm_range_snr,
        lte_mask  = True,
        raise_missing_signal = False,
        do_rfi_clean = True
        )
        print("DM after Downsample2 get_snr()", DM)
        profile = get_profile(power)
        f = old_ds/downsampling_factor
        start, end = int(start*f), int(min(end*f, len(profile)))
        plt.clf()
        if freq_range is None:
            f_start = 0
            f_end = power.shape[0] - 1
        if fit_DM:
            try:
                DM_corr, DM_err = get_structure_max_DM(power[f_start:f_end,start: end], freq, t_res = 2.56e-6 * downsampling_factor, DM_range = 0.3)
                plt.show()
            except Exception as e:
                print("DM_phase failed, skipping it...")
                print(e)
                DM_corr ,DM_err = 0, 0.05/2
        else:
            DM_corr ,DM_err = 0, 0.05/2
        print(dm_range_snr)
        freq_id, freq, power, _, _, valid_channels, _, DM, downsampling_factor = get_snr(
        data,
        DM = DM + DM_corr,
        downsample=downsample2,
        fill_missing_time = fill_missing_time,
        spectrum_lim = spectrum_lim,
        diagnostic_plots=False,
        return_full=True,
        DM_range = dm_range_snr,
        lte_mask  = True,
        raise_missing_signal = False,
        do_rfi_clean = True
        )
        print("DM after downsample2 struc max get_snr function", DM)
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
    
def baseband_processing(path : str, nwalkers : int = 100, nchain : int = 500000, 
    ncores : int = 10, time_range : tuple = None, freq_range : tuple = None, downsample : int = None, 
    downsample2 : int = None, DM : float = None, fit_DM : bool = True,
    logl : int = 0, save : bool = None, spectrum_lim : bool = True, 
    fill_missing_time : bool = None, show_chains : bool = False, 
    peaks : list = None, fit_spectrum = False, diagnostic_plots : bool = False) -> tuple:
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
    fit_spectrum: bool, optional
        If True then perform mcmc fit on the spectrum of each component. Default is False
    diagnostic_plots : bool, optional
        If True, shows plots of many intermediate steps. Default is False.
    Returns
    -------
    Tuple
        All the required data and fit parameters to make the fitburst input file.
    """
    # Run baseband processing
    event_id = path.split('/')[-2].split('_')[-1]
    data, freq_id, freq, power, valid_channels, DM, DM_err, downsampling_factor, profile, t_res, peak_times, start, full_power = read_baseband(path, 
                                diagnostic_plots = diagnostic_plots, time_range = time_range, freq_range = freq_range, 
                                fill_missing_time = fill_missing_time, downsample = downsample, downsample2 = downsample2, 
                                DM = DM, fit_DM = fit_DM, peaks = peaks, spectrum_lim = spectrum_lim, return_full = True)
    # Arrival times = peak times, widths = default value
    return  data, freq_id, freq, power, DM, downsampling_factor, t_res, peak_times, valid_channels 
