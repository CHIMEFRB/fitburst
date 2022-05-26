import numpy as np
from fitburst.backend.baseband import fit_profile

def make_input(path : str, peaks : list = None, nwalkers : int = 50, nchain : int = 500000, ncores : int = 10, 
    show_chains : bool = False, DM : float = None, fit_DM : bool = True, logl : int = 0, 
    save : str = None, diagnostic_plots : bool = False, downsample2 : int = None, 
    downsample : int = None, spectrum_lim : bool = True, fill_missing_time : bool = None, 
    ref_freq : float = 600., fit_spectrum : bool = False, time_range : tuple = None, output : str = None) -> None:
    """
    Produces the fitburst input .npz file with burst data, metadata, and parameters.
    
    Parameters
    ----------
    path : str
        Path to the singlebeam h5 file that contains baseband data.
    peaks : list, optional
        List of bin numbers at downsampling 'downsample' where to put the initial guess
        for the peak position of each component. If not provided, use lowess algorithm.
    nwalkers : int, optional
        Number of walkers in MCMC process, default is 50
    nchain : int, optional
        Length of chain in MCMC process, default is 500k
    ncores : int, optional
        Number of frb-analysis cores that are used for the MCMC fitting, 
        default is 10, in general, do not use more than ~15.
    show_chains : bool, optional
        If true, produce plots of the parameter value vs chain step.
        Default is False.
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
    diagnostic_plots : bool, optional
        If True, shows plots of many intermediate steps. Default is False.
    downsample : int, optional
        To downsample the data in addition to the downsampling in the singlebeam file.
        This downsampling factor is used in the DM correction and in the peak finding steps.
    downsample2 : int, optional
        To downsample the data by a different factor than 'downsample' for all the steps after peak finding.
        This is useful if you want to downsample more to find peaks more easily, but then downsample less to
        make MCMC fitting easier.
    spectrum_lim : bool, optional
        Whether to cut out the frequency channels without strong signal (True) or not. 
        Default is True.
    fill_missing_time : bool, optional
        If you want to force get_snr to apply the procedue to fill the triangular artefact with noise.
        Default is None (i.e. get_snr decides whether to fill the triangle or not). 
    ref_freq : float, optional
        Reference frequency in MHz with respect to which quantities such as arrival-time 
        and power-law parameter estimates are calculated
    time_range : tuple, optional
        Bin numbers where to start and end the baseband data at downsampling 'downsample'.
    output : str, optional
        Path where to save and name of fitburst input file.
        Default is to save in working directory with file name 'fitburst_input_<event_ID>'
    Returns
    -------
    None
        A .npz file with name and path 'output' containing the data, metadata, 
        and initial fitburst parameters is saved to disk.       
    """
    #Run profile analysis to estimate burst parameters
    data, freq_id, freq, power, DM, downsampling_factor, t_res, params, spectrum_pars, valid_channels = fit_profile(path,
                   nwalkers=nwalkers, nchain=nchain, ncores=ncores, downsample2 = downsample2,
                   downsample = downsample, DM = DM, fit_DM = fit_DM, fit_spectrum = fit_spectrum, logl = logl, 
                   peaks = peaks, fill_missing_time = fill_missing_time, save = save, spectrum_lim = spectrum_lim, 
                   time_range = time_range, show_chains=show_chains, diagnostic_plots=diagnostic_plots)
    
    params, scattering_t = np.array(params[:-1]), params[-1]
    n = int(len(params)/3)
    params = params.reshape((n,3))
    amps = params[...,0]
    mus = params[...,1]
    widths = params[...,2]
    
    data_full = np.zeros([1024, power.shape[-1]])
    data_full[1023 - freq_id] = power
    bad = []
    for i in range(1024):
        if i not in 1023 - freq_id:
            bad.append(i)
    #Metadata
    metadata = {
    "bad_chans"      : bad, # a Python list of indices corresponding to frequency channels to zero-weight
    "freqs_bin0"     : min(freq), # a floating-point scalar indicating the value of frequency bin at index 0, in MHz
    "is_dedispersed" : True, # a boolean indicating if spectrum is already dedispersed (True) or not (False)
    "num_freq"       : data_full.shape[0], # an integer scalar indicating the number of frequency bins/channels
    "num_time"       : data_full.shape[-1], # an integer scalar indicating the number of time bins
    "times_bin0"     : 0., # a floating-point scalar indicating the value of time bin at index 0, in seconds
    "res_freq"       : 400./1024., # a floating-point scalar indicating the frequency resolution, in MHz
    "res_time"       : t_res # a floating-point scalar indicating the time resolution, in seconds
    }

    
    #Output has shape burst_parameters, num components
    #Burst parameters
    burst_parameters = {
    "amplitude"            : list(np.log10(amps / np.sqrt(data_full.shape[0] - len(bad)))), # Currently wrong. Should reflect inverse of how profile is calculated
    "arrival_time"         : list(mus), # a list containing the arrival times, in seconds
    "burst_width"          : list(widths), # a list containing the temporal widths, in seconds
    "dm"                   : [DM for i in range(len(mus))], # a list containing the dispersion measures (DM), in parsec per cubic centimeter
    "dm_index"             : [-2 for i in range(len(mus))], # a list containing the exponents of frequency dependence in DM delay
    "ref_freq"             : [ref_freq for i in range(len(mus))], # a list containing the reference frequencies for arrival-time and power-law parameter estimates, in MHz (held fixed)
    "scattering_index"     : [-4 for i in range(len(mus))], # a list containing the exponents of frequency dependence in scatter-broadening
    "scattering_timescale" : [scattering_t for i in range(len(mus))], # a list containing the scattering timescales, in seconds
    "spectral_index"       : list(spectrum_pars[1]), # a list containing the power-law spectral indices
    "spectral_running"     : list(spectrum_pars[2]) # a list containing the power-law spectral running
    }
    

    if output is not None:
        fname = output
    else:
        fname = "fitburst_input_"+event_id
    np.savez(
        fname + ".npz", 
        data_full=data_full, 
        metadata=metadata, 
        burst_parameters=burst_parameters
    )
    return