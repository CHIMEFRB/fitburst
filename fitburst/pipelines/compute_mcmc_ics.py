import numpy as np
from fitburst.pipelines.mcmc_ics import fit_profile

def make_input(path : str, nwalkers : int = 50, nchain : int = 500000, ncores : int = 10, 
    show_chains : bool = False, logl : int = 0, fit_spectrum : bool = False,
    diagnostic_plots : bool = False, event_id : str = None,
    ref_freq : float = 600., output : str = None) -> None:
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
    data, params, spectrum_pars = fit_profile(path,
                   nwalkers=nwalkers, nchain=nchain, ncores=ncores, logl = logl,
                   fit_spectrum = fit_spectrum, show_chains=show_chains, diagnostic_plots=diagnostic_plots)
    
    params, scattering_t = np.array(params[:-1]), params[-1]
    n = int(len(params)/3)
    params = params.reshape((n,3))
    amps = params[...,0]
    mus = params[...,1]
    widths = params[...,2]
    
    #Metadata
    metadata = {
    "bad_chans"      : np.where(data.data_weights == 0), # a Python list of indices corresponding to frequency channels to zero-weight
    "freqs_bin0"     : min(data.freqs), # a floating-point scalar indicating the value of frequency bin at index 0, in MHz
    "is_dedispersed" : True, # a boolean indicating if spectrum is already dedispersed (True) or not (False)
    "num_freq"       : data.num_freq, # an integer scalar indicating the number of frequency bins/channels
    "num_time"       : data.num_time, # an integer scalar indicating the number of time bins
    "times_bin0"     : 0., # a floating-point scalar indicating the value of time bin at index 0, in seconds
    "res_freq"       : 400./1024., # a floating-point scalar indicating the frequency resolution, in MHz
    "res_time"       : data.res_time # a floating-point scalar indicating the time resolution, in seconds
    }

    
    #Output has shape burst_parameters, num components
    #Burst parameters
    burst_parameters = {
    "amplitude"            : list(np.log10(amps / np.sqrt(data.data_full.shape[0] - len(np.where(data.data_weights == 0))))), # Currently wrong. Should reflect inverse of how profile is calculated
    "arrival_time"         : list(mus), # a list containing the arrival times, in seconds
    "burst_width"          : list(widths), # a list containing the temporal widths, in seconds
    "dm"                   : [data.burst_parameters['dm'] for i in range(len(mus))], # a list containing the dispersion measures (DM), in parsec per cubic centimeter
    "dm_index"             : [-2 for i in range(len(mus))], # a list containing the exponents of frequency dependence in DM delay
    "ref_freq"             : [ref_freq for i in range(len(mus))], # a list containing the reference frequencies for arrival-time and power-law parameter estimates, in MHz (held fixed)
    "scattering_index"     : [-4 for i in range(len(mus))], # a list containing the exponents of frequency dependence in scatter-broadening
    "scattering_timescale" : [scattering_t for i in range(len(mus))], # a list containing the scattering timescales, in seconds
    "spectral_index"       : list(spectrum_pars[1]), # a list containing the power-law spectral indices
    "spectral_running"     : list(spectrum_pars[2]) # a list containing the power-law spectral running
    }
    
    if event_id is None:
        event_id = path.split('/')[-1].split('_')[-1].split('.')[0]

    if output is not None:
        fname = output
    else:
        fname = "fitburst_input_"+event_id
    np.savez(
        fname + ".npz", 
        data_full=data.data_full, 
        metadata=metadata, 
        burst_parameters=burst_parameters
    )
    return
