"""
Routines for Partial Derivatives of the fitburst Model w.r.t. Fit Parameters

This module contains functions that return partial derivatives of the model 
defined by fitburt. Each derivative is only defined for fittable parameters, and 
so the fitter object will select which derivatives to compute based on 
fit parameters.
"""

import scipy.special as ss
import numpy as np

def deriv_model_wrt_amplitude(parameters: dict, model: object, component: int = 0) -> float:
    """
    Computes the derivative of the model with respect to the amplitude parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    deriv_mod = np.log(10) * model.spectrum_per_component[:, :, component]

    return deriv_mod


def deriv_model_wrt_spectral_running(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the derivative of the model with respect to the spectral-running parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod =  log_freq[:, None] * model.spectrum_per_component[:, :, component]

    return deriv_mod

def deriv_model_wrt_spectral_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the derivative of the model with respect to the spectral-index parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq)
    deriv_mod = log_freq[:, None] * model.spectrum_per_component[:, :, component]

    return deriv_mod

def deriv_model_wrt_burst_width(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the derivative of the model with respect to the burst-width parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

     # get dimensions and define empty model-derivative matrix.
    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    num_freq, num_time, num_component = model.timediff_per_component.shape
    ref_freq = parameters["ref_freq"][component]
    scattering_index = parameters["scattering_index"][0]
    scattering_timescale = parameters["scattering_timescale"][0]
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]
    timediff = model.timediff_per_component[:, :, component]
    freq_ratio = model.freqs / ref_freq
    scat_times_freq = scattering_timescale * freq_ratio ** scattering_index
    deriv_mod = np.zeros((num_freq, num_time), dtype=float)

    for freq in range(num_freq):
        current_timediff = timediff[freq, :]

        if scat_times_freq[freq] < np.fabs(0.15 * burst_width):
            deriv_mod[freq, :] += (current_timediff ** 2 * model.spectrum_per_component[freq, :, component] 
                / burst_width ** 3)

        else:

            # define argument of error and scattering timescales over frequency.
            log_freq = np.log(freq_ratio[freq])
            spectrum = 10 ** amplitude * freq_ratio[freq] ** (spectral_index + spectral_running * log_freq)
            #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width
            erf_arg = (current_timediff - burst_width ** 2  / scat_times_freq[freq]) / burst_width / np.sqrt(2)
            exp_arg = (burst_width / scat_times_freq[freq]) ** 2 / 2 - current_timediff / scat_times_freq[freq] - erf_arg ** 2

            # now compute derivative contribution from current component.
            term1 = burst_width * model.spectrum_per_component[freq, :, component] / scat_times_freq[freq] ** 2 
            term2 = -np.sqrt(2 / np.pi) * spectrum * current_timediff * np.exp(exp_arg) / \
                    burst_width ** 2 / scat_times_freq[freq] 

            deriv_mod[freq, :] += term1 + term2 

    return deriv_mod

def deriv_model_wrt_arrival_time(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the derivative of the model with respect to the arrival-time parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    num_freq, num_time, num_component = model.timediff_per_component.shape
    deriv_mod = np.zeros((num_freq, num_time), dtype=float)

    # now loop over each model component and compute contribution to derivative.
    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    ref_freq = parameters["ref_freq"][component]
    scattering_index = parameters["scattering_index"][0]
    scattering_timescale = parameters["scattering_timescale"][0]
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]
    timediff = model.timediff_per_component[:, :, component]
    freq_ratio = model.freqs / ref_freq
    scat_times_freq = scattering_timescale * freq_ratio ** scattering_index

    for freq in range(num_freq):
        current_timediff = timediff[freq, :]

        if scat_times_freq[freq] < np.fabs(0.15 * burst_width):
            deriv_mod[freq, :] += current_timediff * model.spectrum_per_component[freq, :, component] / burst_width ** 2

        else:
            # define argument of error and scattering timescales over frequency.a
            log_freq = np.log(freq_ratio[freq])
            spectrum = 10 ** amplitude * freq_ratio[freq] ** (spectral_index + spectral_running * log_freq)
            #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width
            erf_arg = (current_timediff - burst_width ** 2  / scat_times_freq[freq]) / burst_width / np.sqrt(2)
            exp_arg = (burst_width / scat_times_freq[freq]) ** 2 / 2 - current_timediff / scat_times_freq[freq] - erf_arg ** 2

            # now compute derivative contribution from current component.
            term1 = model.spectrum_per_component[freq, :, component] / scat_times_freq[freq]
            term2 = -np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) / burst_width / scat_times_freq[freq]

            deriv_mod[freq, :] += term1 + term2

    return deriv_mod

def deriv_model_wrt_dm(parameters: dict, model: float, component: int = 0, add_all: bool = True) -> float:
    """
    Computes the derivative of the model with respect to the dm parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    num_freq, num_time, num_component = model.timediff_per_component.shape
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    scattering_index = parameters["scattering_index"][0]
    scattering_timescale = parameters["scattering_timescale"][0]

    # now loop over each model component and compute contribution to derivative.
    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]
        timediff = model.timediff_per_component[:, :, current_component]
        freq_ratio = model.freqs / ref_freq
        freq_diff = model.freqs ** dm_index - ref_freq ** dm_index
        scat_times_freq = scattering_timescale * freq_ratio ** scattering_index

        for freq_idx in range(num_freq):
            current_timediff = timediff[freq_idx, :]

            if scat_times_freq[freq_idx] < np.fabs(0.15 * burst_width):
                deriv_timediff_wrt_dm = -dm_const * freq_diff[freq_idx]
                deriv_mod_int[freq_idx, :, current_component] += (-current_timediff * 
                    model.spectrum_per_component[freq_idx, :, current_component] / burst_width ** 2 * deriv_timediff_wrt_dm)

            else:

                # define argument of error and scattering timescales over frequency.
                log_freq = np.log(freq_ratio[freq_idx])
                spectrum = 10 ** amplitude * freq_ratio[freq_idx] ** (spectral_index + spectral_running * log_freq)
                #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width
                erf_arg = (current_timediff - burst_width ** 2  / scat_times_freq[freq_idx]) / burst_width / np.sqrt(2)
                exp_arg = (burst_width / scat_times_freq[freq_idx]) ** 2 / 2 - current_timediff / scat_times_freq[freq_idx] - \
                          erf_arg ** 2

                # now compute derivative contribution from current component.
                term1 = model.spectrum_per_component[freq_idx, :, current_component] / scat_times_freq[freq_idx]
                term2 = -np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) / burst_width / scat_times_freq[freq_idx]
                deriv_mod_int[freq_idx, :, current_component] += dm_const * freq_diff[freq_idx] * (term1 + term2)

    # now determine if all components should be summed or not.
    deriv_mod = deriv_mod_int[:, :, component]

    if add_all:
        deriv_mod = np.sum(deriv_mod_int, axis = 2)

    return deriv_mod

def deriv_model_wrt_dm_index(parameters: dict, model: float, component: int = 0, add_all: bool = True) -> float:
    """
    Computes the derivative of the model with respect to the dm-index parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    deriv_mod_int = np.zeros(model.spectrum_per_component.shape)
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    
    for current_component in range(model.spectrum_per_component.shape[2]):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]


        for freq_idx in range (model.spectrum_per_component.shape[0]):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            sc_time_freq = sc_time * freq_ratio ** sc_index
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            product = np.log(model.freqs[freq_idx]) * model.freqs[freq_idx] ** dm_index -\
                      np.log(ref_freq) * ref_freq ** dm_index

            term1 = dm_const * dm * product * \
                model.spectrum_per_component[freq_idx, :, current_component] / sc_time_freq
            term2 = -np.sqrt(2 / np.pi) * product * spectrum * np.exp(exp_arg) / burst_width / sc_time_freq

            deriv_mod_int[freq_idx, :, current_component] = term1 + term2

    # now determine if all components should be summed or not.
    deriv_mod = deriv_mod_int[:, :, component]

    if add_all:
        deriv_mod = np.sum(deriv_mod_int, axis = 2)

    return deriv_mod 

def deriv_model_wrt_scattering_timescale(parameters: dict, model: float, component: int = 0, add_all: bool = True) -> float:
    """
    Computes the derivative of the model with respect to the scattering-timescale parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    add_all : bool, optional
        If True, then sum all per-component evaluations of first derivative; 
        otherwise, return derivative map for component with index 'component'

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """
 
    # get dimensions and define empty model-derivative matrix.
    num_freq, num_time, num_component = model.timediff_per_component.shape
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    # now loop over each model component and compute contribution to derivative.
    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        ref_freq = parameters["ref_freq"][current_component]
        scattering_index = parameters["scattering_index"][0]
        scattering_timescale = parameters["scattering_timescale"][0]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]
        timediff = model.timediff_per_component[:, :, current_component]
        
        # define argument of error and scattering timescales over frequency.
        freq_ratio = model.freqs / ref_freq
        log_freq = np.log(freq_ratio)
        spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * log_freq)
        #timediff[timediff < -5 * burst_width] = -5 * burst_width
        scat_times_freq = scattering_timescale * freq_ratio ** scattering_index
        erf_arg = (timediff - burst_width ** 2  / scat_times_freq[:, None]) / burst_width / np.sqrt(2)
        product = np.exp((burst_width / scat_times_freq[:, None]) ** 2 / 2 - timediff / scat_times_freq[:, None]
            - erf_arg ** 2)

        # now compute derivative contribution from current component.
        term1 = model.spectrum_per_component[:, :, current_component] 
        term2 = (burst_width / scat_times_freq[:, None]) ** 2 * model.spectrum_per_component[:, :, current_component]
        term3 = -timediff / scat_times_freq[:, None] * model.spectrum_per_component[:, :, current_component]
        term4 = -np.sqrt(2 / np.pi) * product * burst_width / scat_times_freq[:, None] ** 2 

        deriv_mod_int[:, :, current_component] = -(term1 + term2 + term3 + term4 * spectrum[:, None]) / scattering_timescale

    # now determine if all components should be summed or not.
    deriv_mod = deriv_mod_int[:, :, component]

    if add_all:
        deriv_mod = np.sum(deriv_mod_int, axis = 2)

    return deriv_mod

def deriv_model_wrt_scattering_index(parameters: dict, model: float, component: int = 0, add_all: bool = True) -> float:
    """
    Computes the derivative of the model with respect to the scattering-index parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    add_all : bool, optional
        If True, then sum all per-component evaluations of first derivative; 
        otherwise, return derivative map for component with index 'component'

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    num_freq, num_time, num_component = model.timediff_per_component.shape
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    # now loop over each model component and compute contribution to derivative.
    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        ref_freq = parameters["ref_freq"][current_component]
        scattering_index = parameters["scattering_index"][0]
        scattering_timescale = parameters["scattering_timescale"][0]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]
        timediff = model.timediff_per_component[:, :, current_component]

        # define argument of error and scattering timescales over frequency.
        freq_ratio = model.freqs / ref_freq
        log_freq = np.log(freq_ratio)
        spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * log_freq)
        #timediff[timediff < -5 * burst_width] = -5 * burst_width
        scat_times_freq = scattering_timescale * freq_ratio ** scattering_index
        erf_arg = (timediff - burst_width ** 2  / scat_times_freq[:, None]) / burst_width / np.sqrt(2)
        product = np.exp((burst_width / scat_times_freq[:, None]) ** 2 / 2 - timediff / scat_times_freq[:, None]
            - erf_arg ** 2)

        # now compute derivative contribution from current component.
        term1 = model.spectrum_per_component[:, :, current_component]
        term2 = (burst_width / scat_times_freq[:, None]) ** 2 * model.spectrum_per_component[:, :, current_component]
        term3 = -timediff / scat_times_freq[:, None] * model.spectrum_per_component[:, :, current_component]
        term4 = -np.sqrt(2 / np.pi) * product * burst_width / scat_times_freq[:, None] ** 2

        deriv_mod_int[:, :, current_component] = -(term1 + term2 + term3 + term4 * spectrum[:, None]) * log_freq[:, None]

    # now determine if all components should be summed or not.
    deriv_mod = deriv_mod_int[:, :, component]

    if add_all:
        deriv_mod = np.sum(deriv_mod_int, axis = 2)

    return deriv_mod

def deriv2_model_wrt_amplitude_amplitude(parameters: dict, model: object, component: int = 0) -> float:
    """
    Computes the second derivative of the model with respect to the amplitude parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The second derivative of the model evaluated over time and frequency
    """

    deriv_mod = np.log(10) ** 2 * model.spectrum_per_component[:, :, component]
    
    return deriv_mod

def deriv2_model_wrt_amplitude_spectral_running(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the amplitude and 
    spectral-running parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod = np.log(10) * log_freq[:, None] * model.spectrum_per_component[:, :, component]

    return deriv_mod

def deriv2_model_wrt_amplitude_spectral_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the amplitude and 
    spectral-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq)
    deriv_mod = np.log(10) * log_freq[:, None] * model.spectrum_per_component[:, :, component]

    return deriv_mod

def deriv2_model_wrt_amplitude_burst_width(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the amplitude and 
    burst-width parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

     # get dimensions and define empty model-derivative matrix.
    deriv_mod = np.log(10) * deriv_model_wrt_burst_width(parameters, model, component)

    return deriv_mod

def deriv2_model_wrt_amplitude_arrival_time(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the amplitude and 
    arrival-time parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

     # get dimensions and define empty model-derivative matrix.
    deriv_mod = np.log(10) * deriv_model_wrt_arrival_time(parameters, model, component)

    return deriv_mod

def deriv2_model_wrt_amplitude_dm(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the amplitude and 
    DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

     # get dimensions and define empty model-derivative matrix.
    deriv_mod = np.log(10) * deriv_model_wrt_dm(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_amplitude_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the amplitude and 
    DM-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

     # get dimensions and define empty model-derivative matrix.
    deriv_mod = np.log(10) * deriv_model_wrt_dm_index(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_amplitude_scattering_timescale(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the amplitude and 
    scattering_timescale parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

     # get dimensions and define empty model-derivative matrix.
    deriv_mod = np.log(10) * deriv_model_wrt_scattering_timescale(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_amplitude_scattering_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the amplitude and 
    scattering-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

     # get dimensions and define empty model-derivative matrix.
    deriv_mod = np.log(10) * deriv_model_wrt_scattering_index(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_spectral_running_spectral_running(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the second partial derivative of the model with respect to the spectral-running parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 4
    deriv_mod =  log_freq[:, None] * model.spectrum_per_component[:, :, component]

    return deriv_mod

def deriv2_model_wrt_spectral_running_spectral_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-running and 
    spectral-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 3
    deriv_mod = log_freq[:, None] * model.spectrum_per_component[:, :, component]

    return deriv_mod

def deriv2_model_wrt_spectral_running_burst_width(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-running and 
    burst-width parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod = log_freq[:, None] * deriv_model_wrt_burst_width(parameters, model, component)

    return deriv_mod

def deriv2_model_wrt_spectral_running_arrival_time(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-running and 
    arrival-time parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod = log_freq[:, None] * deriv_model_wrt_arrival_time(parameters, model, component)

    return deriv_mod

def deriv2_model_wrt_spectral_running_dm(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-running and 
    DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod = log_freq[:, None] * deriv_model_wrt_dm(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_spectral_running_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-running and 
    DM-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod = log_freq[:, None] * deriv_model_wrt_dm_index(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_spectral_running_scattering_timescale(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-running and 
    scattering-timescale parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod = log_freq[:, None] * deriv_model_wrt_scattering_timescale(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_spectral_running_scattering_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-running and 
    scattering-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod = log_freq[:, None] * deriv_model_wrt_scattering_index(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_spectral_index_spectral_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the second partial derivative of the model with respect to the spectral-index parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) ** 2
    deriv_mod = log_freq[:, None] * model.spectrum_per_component[:, :, component]

    return deriv_mod

def deriv2_model_wrt_spectral_index_burst_width(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-index and 
    burst-width parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq)
    deriv_mod = log_freq[:, None] * deriv_model_wrt_burst_width(parameters, model, component)

    return deriv_mod

def deriv2_model_wrt_spectral_index_arrival_time(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-index and 
    burst-width parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq) 
    deriv_mod = log_freq[:, None] * deriv_model_wrt_arrival_time(parameters, model, component)

    return deriv_mod

def deriv2_model_wrt_spectral_index_dm(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-index and 
    DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq)
    deriv_mod = log_freq[:, None] * deriv_model_wrt_dm(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_spectral_index_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-index and 
    DM-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq)
    deriv_mod = log_freq[:, None] * deriv_model_wrt_dm_index(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_spectral_index_scattering_timescale(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-index and 
    scattering-timescale parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq)
    deriv_mod = log_freq[:, None] * deriv_model_wrt_scattering_timescale(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_spectral_index_scattering_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the spectral-index and 
    scattering-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """

    ref_freq = parameters["ref_freq"][component]
    log_freq = np.log(model.freqs / ref_freq)
    deriv_mod = log_freq[:, None] * deriv_model_wrt_scattering_index(parameters, model, component, add_all = False)

    return deriv_mod

def deriv2_model_wrt_burst_width_burst_width(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the second partial derivative of the model with respect to the burst-width parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The second derivative of the model evaluated over time and frequency
    """

    # define parameters and objects needed for mixed-derivative calculation.
    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_burst_width(parameters, model, component)
    deriv_mod = np.zeros(current_model.shape)
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        sc_time_freq = sc_time * freq_ratio ** sc_index

        # if scattering is not resolvable, then assume Gaussian temporal profile.
        if sc_time_freq < np.fabs(0.15 * burst_width):
            deriv_mod[freq_idx, :] = -3 * current_timediff[freq_idx, :] ** 2 * \
                                     current_model[freq_idx, :] / burst_width ** 4
            deriv_mod[freq_idx, :] += (current_timediff[freq_idx, :] ** 4 * \
                                      current_model[freq_idx, :] / burst_width ** 6)

        else:
            # adjust time-difference values to make them friendly for error function.
            #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            
            erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
            erf_arg_deriv = -(current_timediff[freq_idx, :] / burst_width ** 2 + 1 / sc_time_freq) / np.sqrt(2)
            exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                      current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = burst_width / sc_time_freq ** 2 - 2 * erf_arg * erf_arg_deriv

            # now define terms that contribute to mixed derivative.
            term1 = current_model[freq_idx, :] / sc_time_freq ** 2
            term2 = burst_width * deriv_first[freq_idx, :] / sc_time_freq ** 2
            term3 = 2 * np.sqrt(2) * current_timediff[freq_idx, :] * spectrum * np.exp(exp_arg) /\
                    sc_time_freq / burst_width ** 3 / np.sqrt(np.pi)
            term4 = -np.sqrt(2 / np.pi) * current_timediff[freq_idx, :] * spectrum * np.exp(exp_arg) *\
                    exp_arg_deriv / sc_time_freq / burst_width ** 2

            deriv_mod[freq_idx, :] = term1 + term2 + term3 + term4

    return deriv_mod

def deriv2_model_wrt_burst_width_arrival_time(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the burst-width and 
    arrival-time parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # define parameters and objects needed for mixed-derivative calculation.
    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_arrival_time(parameters, model, component)
    deriv_mod = np.zeros(current_model.shape)
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        sc_time_freq = sc_time * freq_ratio ** sc_index

        # if scattering is not resolvable, then assume Gaussian temporal profile.
        if sc_time_freq < np.fabs(0.15 * burst_width):
            deriv_mod[freq_idx, :] = -2 * current_timediff[freq_idx, :] * current_model[freq_idx, :] / burst_width ** 3
            deriv_mod[freq_idx, :] += (current_timediff[freq_idx, :] ** 3 * current_model[freq_idx, :] / burst_width ** 5)

        else:
            # adjust time-difference values to make them friendly for error function.
            #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
            erf_arg_deriv = -1 / burst_width / np.sqrt(2)
            exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                      current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = 1 / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            # now define terms that contriubte to mixed-partial derivative.
            term1 = burst_width * deriv_first[freq_idx, :] / sc_time_freq ** 2
            term2 = np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) / burst_width ** 2 / sc_time_freq
            term3 = -np.sqrt(2 / np.pi) * current_timediff[freq_idx, :] * spectrum * np.exp(exp_arg) *\
                    exp_arg_deriv / burst_width ** 2 / sc_time_freq 

            deriv_mod[freq_idx, :] = term1 + term2 + term3

    return deriv_mod

def deriv2_model_wrt_burst_width_scattering_timescale(parameters: dict, model: float, component: int = 0):
    """
    Computes the mixed partial derivative of the model with respect to the burst-width and 
    scattering-timescale parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # define parameters and objects needed for mixed-derivative calculation.
    amplitude = parameters["amplitude"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_scattering_timescale(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape)
    burst_width = parameters["burst_width"][component]
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]

    # adjust time-difference values to make them friendly for error function.
    #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        freq_ratio_sc = freq_ratio ** sc_index
        sc_time_freq = sc_time * freq_ratio_sc
        spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
        erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
        erf_arg_deriv = burst_width * freq_ratio_sc / sc_time_freq ** 2 / np.sqrt(2)
        exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                  current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
        exp_arg_deriv = -burst_width ** 2 * freq_ratio_sc / sc_time_freq ** 3 + \
                        current_timediff[freq_idx, :] / sc_time_freq ** 2 * freq_ratio_sc - \
                        2 * erf_arg * erf_arg_deriv

        # now define terms that contriubte to mixed-partial derivative.
        term1 = -2 * burst_width * current_model[freq_idx, :] * freq_ratio_sc / sc_time_freq ** 3
        term2 = burst_width * deriv_first[freq_idx, :] / sc_time_freq ** 2
        term3 = np.sqrt(2 / np.pi) * current_timediff[freq_idx, :] * freq_ratio_sc * spectrum * np.exp(exp_arg) /\
                burst_width ** 2 / sc_time_freq ** 2
        term4 = -np.sqrt(2 / np.pi) * current_timediff[freq_idx, :] * spectrum * np.exp(exp_arg) * exp_arg_deriv /\
                burst_width ** 2 / sc_time_freq

        deriv_mod[freq_idx, :] = term1 + term2 + term3 + term4

    return deriv_mod

def deriv2_model_wrt_burst_width_scattering_index(parameters: dict, model: float, component: int = 0):
    """
    Computes the mixed partial derivative of the model with respect to the burst-width and 
    scattering-timescale parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # define parameters and objects needed for mixed-derivative calculation.
    amplitude = parameters["amplitude"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_scattering_index(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape)
    burst_width = parameters["burst_width"][component]
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]

    # adjust time-difference values to make them friendly for error function.
    #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        freq_ratio_sc = freq_ratio ** sc_index
        sc_time_freq = sc_time * freq_ratio_sc
        spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))

        erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
        erf_arg_deriv = burst_width * np.log(freq_ratio) / sc_time_freq / np.sqrt(2)
        exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                  current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
        exp_arg_deriv = -(burst_width / sc_time_freq) ** 2 * np.log(freq_ratio) + \
                        current_timediff[freq_idx, :] / sc_time_freq * np.log(freq_ratio) - 2 * erf_arg * erf_arg_deriv

        # now define terms that contriubte to mixed-partial derivative.
        term1 = -2 * burst_width * current_model[freq_idx, :] * np.log(freq_ratio) / sc_time_freq ** 2
        term2 = burst_width * deriv_first[freq_idx, :] / sc_time_freq ** 2
        term3 = np.sqrt(2 / np.pi) * current_timediff[freq_idx, :] * np.log(freq_ratio) * spectrum * np.exp(exp_arg) /\
                burst_width ** 2 / sc_time_freq
        term4 = -np.sqrt(2 / np.pi) * current_timediff[freq_idx, :] * spectrum * np.exp(exp_arg) * exp_arg_deriv /\
                burst_width ** 2 / sc_time_freq

        deriv_mod[freq_idx, :] = term1 + term2 + term3 + term4

    return deriv_mod

def deriv2_model_wrt_burst_width_dm(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the burst-width and 
    DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_dm(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape)
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component] # global parameter.
    spectral_running = parameters["spectral_running"][component] # global parameter.

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        product = dm_const * (model.freqs[freq_idx] ** dm_index - ref_freq ** dm_index)
        sc_time_freq = sc_time * freq_ratio ** sc_index

        # if scattering is not resolvable, then assume Gaussian temporal profile.
        if sc_time_freq < np.fabs(0.15 * burst_width):
            deriv_mod[freq_idx, :] += -2 * current_timediff[freq_idx, :] * product * \
                                      current_model[freq_idx, :] / burst_width ** 3
            deriv_mod[freq_idx, :] += (current_timediff[freq_idx, :] ** 2 * deriv_first[freq_idx, :] / burst_width ** 3)

        else:
            # adjust time-difference values to make them friendly for error function.
            #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
            erf_arg_deriv = -product / burst_width / np.sqrt(2)
            exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                      current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = product / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            # now define terms that contriubte to mixed-partial derivative.
            term1 = burst_width * deriv_first[freq_idx, :] / sc_time_freq ** 2
            term2 = np.sqrt(2 / np.pi) * spectrum * product * np.exp(exp_arg) / burst_width ** 2 / sc_time_freq
            term3 = -np.sqrt(2 / np.pi) * current_timediff[freq_idx, :] * spectrum * np.exp(exp_arg) *\
                    exp_arg_deriv / burst_width ** 2 / sc_time_freq

            deriv_mod[freq_idx, :] = term1 + term2 + term3

    return deriv_mod

def deriv2_model_wrt_burst_width_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the burst-width and 
    DM-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_dm_index(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape)
    dm = parameters["dm"][0] # global parameter.
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    dm_index = parameters["dm_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    sc_index = parameters["scattering_index"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component] # global parameter.
    spectral_running = parameters["spectral_running"][component] # global parameter.

    # adjust time-difference values to make them friendly for error function.
    #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        product = dm_const * dm * (np.log(model.freqs[freq_idx]) * model.freqs[freq_idx] ** dm_index - \
                  np.log(ref_freq) * ref_freq ** dm_index)
        sc_time_freq = sc_time * freq_ratio ** sc_index

        # if scattering is not resolvable, then assume Gaussian temporal profile.
        if sc_time_freq < np.fabs(0.15 * burst_width):
            deriv_mod[freq_idx, :] += -2 * current_timediff[freq_idx, :] * product * current_model[freq_idx, :] / burst_width ** 3
            deriv_mod[freq_idx, :] += (current_timediff[freq_idx, :] ** 2 * deriv_first[freq_idx, :] / burst_width ** 3)
   
        else:
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
            erf_arg_deriv = -product / burst_width / np.sqrt(2)
            exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                      current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = product / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            # now define terms that contriubte to mixed-partial derivative.
            term1 = burst_width * deriv_first[freq_idx, :] / sc_time_freq ** 2
            term2 = np.sqrt(2 / np.pi) * spectrum * product * np.exp(exp_arg) / burst_width ** 2 / sc_time_freq
            term3 = -np.sqrt(2 / np.pi) * current_timediff[freq_idx, :] * spectrum * np.exp(exp_arg) *\
                    exp_arg_deriv / burst_width ** 2 / sc_time_freq

            deriv_mod[freq_idx, :] = term1 + term2 + term3
 
    return deriv_mod

def deriv2_model_wrt_arrival_time_arrival_time(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the second partial derivative of the model with respect to the arrival-time parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_arrival_time(parameters, model, component)
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]

    deriv_mod = np.zeros((num_freq, num_time), dtype=float)

    for freq_idx in range(num_freq):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        sc_time_freq = sc_time * freq_ratio ** sc_index

        if sc_time_freq < np.fabs(0.15 * burst_width):
            deriv_mod[freq_idx, :] = current_timediff[freq_idx, :] ** 2 * current_model[freq_idx, :] / burst_width ** 4
            deriv_mod[freq_idx, :] -= (current_model[freq_idx, :] / burst_width ** 2)

        else:
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -1 / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = (1 / sc_time_freq - 2 * erf_arg * erf_arg_deriv)

            term1 = deriv_first[freq_idx, :] / sc_time_freq
            term2 = -np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) * exp_arg_deriv / burst_width / sc_time_freq
            deriv_mod[freq_idx, :] = term1 + term2

    return deriv_mod

def deriv2_model_wrt_arrival_time_dm(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the arrival-time and 
    DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_dm(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape, dtype=float)
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        sc_time_freq = sc_time * freq_ratio ** sc_index

        if sc_time_freq < np.fabs(0.15 * burst_width):
            deriv_mod[freq_idx, :] = current_timediff[freq_idx, :] * deriv_first[freq_idx, :] / burst_width ** 2
            deriv_mod[freq_idx, :] -= (dm_const * (model.freqs[freq_idx] ** dm_index - ref_freq ** dm_index) * 
                                      current_model[freq_idx, :] / burst_width ** 2)

        else:
            # adjust time-difference values to make them friendly for error function.
            #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
            erf_arg_deriv = -dm_const * (model.freqs[freq_idx] ** dm_index - ref_freq ** dm_index) / burst_width / np.sqrt(2)
            exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                      current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = dm_const * (model.freqs[freq_idx] ** dm_index - ref_freq ** dm_index) / sc_time_freq - \
                            2 * erf_arg * erf_arg_deriv

            # now define terms that contriubte to mixed-partial derivative.
            term1 = deriv_first[freq_idx, :] / sc_time_freq
            term2 = -np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) * exp_arg_deriv / burst_width / sc_time_freq
            deriv_mod[freq_idx, :] = term1 + term2

    return deriv_mod

def deriv2_model_wrt_arrival_time_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the arrival-time and 
    DM-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    dm = parameters["dm"][0] # global parameter.
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]

    deriv_mod = current_timediff * deriv_model_wrt_dm_index(parameters, model, component) / burst_width ** 2
    deriv_mod += (dm_const * dm * (np.log(model.freqs[:, None]) * model.freqs[:, None] ** dm_index - \
                 np.log(ref_freq) * ref_freq ** dm_index) * current_model / burst_width ** 2)

    return deriv_mod

def deriv2_model_wrt_arrival_time_scattering_timescale(parameters: dict, model: float, component: int = 0):
    """
    Computes the mixed partial derivative of the model with respect to the arrival-time and 
    scattering-timescale parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_scattering_timescale(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape)
    dm = parameters["dm"][0] # global parameter.
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    dm_index = parameters["dm_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    sc_index = parameters["scattering_index"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component] # global parameter.
    spectral_running = parameters["spectral_running"][component] # global parameter.

    # adjust time-difference values to make them friendly for error function.
    #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        sc_time_freq = sc_time * freq_ratio ** sc_index
        spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
        erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
        erf_arg_deriv = burst_width * freq_ratio ** sc_index / sc_time_freq ** 2 / np.sqrt(2)
        exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                  current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
        exp_arg_deriv = -burst_width ** 2 * freq_ratio ** sc_index / sc_time_freq ** 3 + \
                        current_timediff[freq_idx, :] * freq_ratio ** sc_index / sc_time_freq ** 2 - \
                        2 * erf_arg * erf_arg_deriv

        # now define terms that contriubte to mixed-partial derivative.
        term1 = -model.spectrum_per_component[freq_idx, :, component] * freq_ratio ** sc_index / sc_time_freq ** 2
        term2 = deriv_first[freq_idx, :] / sc_time_freq
        term3 = np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) * freq_ratio ** sc_index / burst_width / sc_time_freq ** 2
        term4 = -np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) * exp_arg_deriv / burst_width / sc_time_freq
        deriv_mod[freq_idx, :] = term1 + term2 + term3 + term4

    return deriv_mod

def deriv2_model_wrt_arrival_time_scattering_index(parameters: dict, model: float, component: int = 0):
    """
    Computes the mixed partial derivative of the model with respect to the arrival-time and 
    scattering-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    amplitude = parameters["amplitude"][component]
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_first = deriv_model_wrt_scattering_index(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape)
    dm = parameters["dm"][0] # global parameter.
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    dm_index = parameters["dm_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    sc_index = parameters["scattering_index"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component] # global parameter.
    spectral_running = parameters["spectral_running"][component] # global parameter.

    # adjust time-difference values to make them friendly for error function.
    #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        sc_time_freq = sc_time * freq_ratio ** sc_index
        spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
        erf_arg = (current_timediff[freq_idx, :] - burst_width ** 2 / sc_time_freq) / burst_width / np.sqrt(2)
        erf_arg_deriv = burst_width * np.log(freq_ratio) / sc_time_freq / np.sqrt(2)
        exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - \
                  current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
        exp_arg_deriv = -burst_width ** 2 * np.log(freq_ratio) / sc_time_freq ** 2 + \
                        current_timediff[freq_idx, :] * np.log(freq_ratio) / sc_time_freq - \
                        2 * erf_arg * erf_arg_deriv

        # now define terms that contriubte to mixed-partial derivative.
        term1 = -model.spectrum_per_component[freq_idx, :, component] * np.log(freq_ratio) / sc_time_freq
        term2 = deriv_first[freq_idx, :] / sc_time_freq
        term3 = np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) * np.log(freq_ratio) / burst_width / sc_time_freq
        term4 = -np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) * exp_arg_deriv / burst_width / sc_time_freq
        deriv_mod[freq_idx, :] = term1 + term2 + term3 + term4

    return deriv_mod

def deriv2_model_wrt_dm_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the arrival-time and 
    DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]

    deriv_mod = dm_const * (model.freqs[:, None] ** dm_index - ref_freq ** dm_index) * current_timediff *\
                deriv_model_wrt_dm_index(parameters, model, component) / burst_width ** 2
    deriv_mod += dm_const ** 2 * dm * current_model / burst_width ** 2 * (model.freqs[:, None] ** dm_index - \
                 ref_freq ** 2) * (np.log(model.freqs[:, None]) * model.freqs[:, None] ** dm_index - \
                 np.log(ref_freq) * ref_freq ** dm_index)
    deriv_mod -= dm_const * (np.log(model.freqs[:, None]) * model.freqs[:, None] ** dm_index - \
                 np.log(ref_freq) * ref_freq ** dm_index) * current_timediff * current_model / burst_width ** 2

    return deriv_mod
   
def deriv2_model_wrt_dm_dm(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the arrival-time and 
    DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    deriv_mod_int = np.zeros(model.spectrum_per_component.shape, dtype=float)
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]

    for current_component in range(model.spectrum_per_component.shape[2]):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_model = model.spectrum_per_component[:, :, current_component]
        current_timediff = model.timediff_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_dm(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq_idx in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq_idx] / ref_freq
        freq_diff = model.freqs[freq_idx] ** dm_index - ref_freq ** dm_index
        sc_time_freq = sc_time * freq_ratio ** sc_index

        if sc_time_freq < np.fabs(0.15 * burst_width):
            deriv_mod_int[freq_idx, :, current_component] = (dm_const * freq_diff) ** 2 * current_timediff[freq_idx, :] ** 2 * \
                current_model[freq_idx, :] / burst_width ** 4
            deriv_mod_int[freq_idx, :, current_component] -= (dm_const * freq_diff) ** 2 * current_model[freq_idx, :] / \
                burst_width ** 2

        else:
            # adjust time-difference values to make them friendly for error function.
            #current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -dm_const * freq_diff / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            exp_arg_deriv = dm_const * freq_diff / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = dm_const * freq_diff * deriv_first[freq_idx, :] / sc_time_freq
            term2 = -dm_const * freq_diff * np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) / burst_width / \
                    sc_time_freq * exp_arg_deriv
            deriv_mod_int[freq_idx, :, current_component] = term1 + term2

    return np.sum(deriv_mod_int, axis = 2)

def deriv2_model_wrt_scattering_timescale_scattering_timescale(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the second partial derivative of the model with respect to the scattering_timescale parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        deriv_first = deriv_model_wrt_dm_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

        for freq_idx in range(num_freq):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = (-1 / sc_time - burst_width ** 2 / sc_time_freq ** 2 / sc_time + current_timediff[freq_idx, :] / \
                    sc_time / sc_time_freq)
            term0_deriv = (1 / sc_time ** 2 + 3 * burst_width ** 2 / (sc_time * sc_time_freq) ** 2 - 
                          2 * current_timediff[freq_idx, :] / sc_time ** 2 / sc_time_freq)
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = burst_width / sc_time_freq / sc_time / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            exp_arg_deriv = -(burst_width / sc_time_freq )** 2 / sc_time + current_timediff[freq_idx, :] / sc_time_freq / sc_time - \
                            2 * erf_arg * erf_arg_deriv


            term1 = term0_deriv * model.spectrum_per_component[freq_idx, :, current_component]
            term2 = term0 * deriv_first[freq_idx, :]
            term3 = -np.sqrt(8 / np.pi) * spectrum * burst_width * np.exp(exp_arg) / sc_time_freq / sc_time ** 2 
            term4 = np.sqrt(2 / np.pi) * spectrum * burst_width * np.exp(exp_arg) * exp_arg_deriv / sc_time_freq / sc_time

            deriv_mod_int[freq_idx, :, current_component] = term1 + term2 + term3 + term4

    return np.sum(deriv_mod_int, axis = 2)

def deriv2_model_wrt_scattering_timescale_scattering_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the second partial derivative of the model with respect to the scattering_timescale parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        deriv_first = deriv_model_wrt_scattering_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

        for freq_idx in range(num_freq):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = (-1 / sc_time - burst_width ** 2 / sc_time_freq ** 2 / sc_time + current_timediff[freq_idx, :] / \
                    sc_time / sc_time_freq)
            term0_deriv = 2 * (burst_width / sc_time_freq) ** 2 / sc_time * np.log(freq_ratio) - \
                          current_timediff[freq_idx, :] * np.log(freq_ratio) / sc_time_freq / sc_time
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = burst_width * np.log(freq_ratio) / sc_time_freq / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            exp_arg_deriv = -(burst_width / sc_time_freq) ** 2 * np.log(freq_ratio) + current_timediff[freq_idx, :] * \
                            np.log(freq_ratio) / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = term0_deriv * model.spectrum_per_component[freq_idx, :, current_component]
            term2 = term0 * deriv_first[freq_idx, :]
            term3 = -np.sqrt(2 / np.pi) * spectrum * burst_width * np.log(freq_ratio) * np.exp(exp_arg) / sc_time_freq / sc_time
            term4 = np.sqrt(2 / np.pi) * spectrum * burst_width * np.exp(exp_arg) * exp_arg_deriv / sc_time_freq / sc_time

            deriv_mod_int[freq_idx, :, current_component] = term1 + term2 + term3 + term4

    return np.sum(deriv_mod_int, axis = 2)

def deriv2_model_wrt_scattering_timescale_dm(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the scattering_timescale 
    and DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_model = model.spectrum_per_component[:, :, current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        deriv_first = deriv_model_wrt_scattering_timescale(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

        for freq_idx in range(num_freq):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            freq_diff = model.freqs[freq_idx] ** dm_index - ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = burst_width / sc_time / sc_time_freq / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = -1 / sc_time * (burst_width / sc_time_freq) ** 2 + current_timediff[freq_idx, :] / sc_time / sc_time_freq \
                            -2 * erf_arg * erf_arg_deriv

            term1 = dm_const * freq_diff * deriv_first[freq_idx, :] / sc_time_freq
            term2 = -dm_const * freq_diff * current_model[freq_idx, :] / sc_time / sc_time_freq
            term3 = dm_const * freq_diff * np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) / burst_width / sc_time_freq / sc_time
            term4 = -dm_const * freq_diff * np.sqrt(2 / np.pi) * spectrum * np.exp(exp_arg) * exp_arg_deriv / burst_width / sc_time_freq

            deriv_mod_int[freq_idx, :, current_component] = term1 + term2 + term3 + term4

    return np.sum(deriv_mod_int, axis = 2)

def deriv2_model_wrt_scattering_timescale_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the scattering_timescale 
    and DM-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        deriv_first = deriv_model_wrt_dm_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

        for freq_idx in range(num_freq):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            freq_diff = np.log(model.freqs[freq_idx]) * model.freqs[freq_idx] ** dm_index - \
                        np.log(ref_freq) * ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = (-1 / sc_time - burst_width ** 2 / sc_time_freq ** 2 / sc_time + current_timediff[freq_idx, :] / \
                    sc_time / sc_time_freq)
            term0_deriv = -dm_const * dm * freq_diff / sc_time_freq / sc_time
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -dm_const * dm * freq_diff / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            exp_arg_deriv = dm_const * dm * freq_diff / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = term0_deriv * model.spectrum_per_component[freq_idx, :, current_component]
            term2 = term0 * deriv_first[freq_idx, :]
            term3 = np.sqrt(2 / np.pi) * spectrum * burst_width * np.exp(exp_arg) * exp_arg_deriv / sc_time_freq / sc_time

            deriv_mod_int[freq_idx, :, current_component] = term1 + term2 + term3

    return np.sum(deriv_mod_int, axis = 2)

def deriv2_model_wrt_scattering_index_scattering_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the second partial derivative of the model with respect to the scattering-index parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        deriv_first = deriv_model_wrt_scattering_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

        for freq_idx in range(num_freq):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            freq_diff = np.log(model.freqs[freq_idx]) * model.freqs[freq_idx] ** dm_index - \
                        np.log(ref_freq) * ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = -(1 + burst_width ** 2 / sc_time_freq ** 2 - current_timediff[freq_idx, :] / sc_time_freq) * np.log(freq_ratio)
            term0_deriv = -np.log(freq_ratio) * (-2 * np.log(freq_ratio) * (burst_width / sc_time_freq) ** 2 + \
                          current_timediff[freq_idx, :] * np.log(freq_ratio) / sc_time_freq)
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = burst_width * np.log(freq_ratio) / sc_time_freq / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            exp_arg_deriv = -np.log(freq_ratio) * (burst_width / sc_time_freq) ** 2 + \
                            current_timediff[freq_idx, :] * np.log(freq_ratio) / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = term0_deriv * model.spectrum_per_component[freq_idx, :, current_component]
            term2 = term0 * deriv_first[freq_idx, :]
            term3 = -np.sqrt(2 / np.pi) * spectrum * burst_width * np.exp(exp_arg) * np.log(freq_ratio) ** 2 / sc_time_freq 
            term4 = np.sqrt(2 / np.pi) * spectrum * burst_width * np.exp(exp_arg) * np.log(freq_ratio) * exp_arg_deriv / sc_time_freq 

            deriv_mod_int[freq_idx, :, current_component] = term1 + term2 + term3 + term4

    return np.sum(deriv_mod_int, axis = 2)

def deriv2_model_wrt_scattering_index_dm(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the scattering-index 
    and DM parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        deriv_first = deriv_model_wrt_dm(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

        for freq_idx in range(num_freq):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            freq_diff = model.freqs[freq_idx] ** dm_index - ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = -(1 + burst_width ** 2 / sc_time_freq ** 2 - current_timediff[freq_idx, :] / sc_time_freq) * np.log(freq_ratio)
            term0_deriv = -np.log(freq_ratio) * dm_const * freq_diff / sc_time_freq
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -dm_const * freq_diff / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            exp_arg_deriv = dm_const * freq_diff / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = term0_deriv * model.spectrum_per_component[freq_idx, :, current_component]
            term2 = term0 * deriv_first[freq_idx, :]
            term3 = np.sqrt(2 / np.pi) * spectrum * burst_width * np.exp(exp_arg) * np.log(freq_ratio) * exp_arg_deriv / sc_time_freq 
            deriv_mod_int[freq_idx, :, current_component] = term1 + term2 + term3

    return np.sum(deriv_mod_int, axis = 2)

def deriv2_model_wrt_scattering_index_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the mixed partial derivative of the model with respect to the scattering-index 
    and DM-index parameters.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        deriv_first = deriv_model_wrt_dm_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

        for freq_idx in range(num_freq):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            freq_diff = np.log(model.freqs[freq_idx]) * model.freqs[freq_idx] ** dm_index - \
                        np.log(ref_freq) * ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = -(1 + burst_width ** 2 / sc_time_freq ** 2 - current_timediff[freq_idx, :] / sc_time_freq) * np.log(freq_ratio)
            term0_deriv = -np.log(freq_ratio) * dm_const * dm * freq_diff / sc_time_freq
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -dm_const * dm * freq_diff / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            exp_arg_deriv = dm_const * dm * freq_diff / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = term0_deriv * model.spectrum_per_component[freq_idx, :, current_component]
            term2 = term0 * deriv_first[freq_idx, :]
            term3 = np.sqrt(2 / np.pi) * spectrum * burst_width * np.exp(exp_arg) * np.log(freq_ratio) * exp_arg_deriv / sc_time_freq 
            deriv_mod_int[freq_idx, :, current_component] = term1 + term2 + term3

    return np.sum(deriv_mod_int, axis = 2)

def deriv2_model_wrt_dm_index_dm_index(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the second partial derivative of the model with respect to the DM-index parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
        The mixed derivative of the model evaluated over time and frequency
    """

    # get dimensions and define empty model-derivative matrix.
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)


    for current_component in range(num_component):
        amplitude = parameters["amplitude"][current_component]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        deriv_first = deriv_model_wrt_dm_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        spectral_index = parameters["spectral_index"][current_component]
        spectral_running = parameters["spectral_running"][current_component]

        for freq_idx in range(num_freq):
            freq_ratio = model.freqs[freq_idx] / ref_freq
            sc_time_freq = sc_time * freq_ratio ** sc_index
            erf_arg = (current_timediff[freq_idx, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq_idx, :] / sc_time_freq - erf_arg ** 2
            spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * np.log(freq_ratio))
            product = dm_const * dm * (np.log(model.freqs[freq_idx]) * model.freqs[freq_idx] ** dm_index -\
                      np.log(ref_freq) * ref_freq ** dm_index)
            product_deriv = dm_const * dm * (np.log(model.freqs[freq_idx]) ** 2 * model.freqs[freq_idx] ** dm_index -\
                            np.log(ref_freq) ** 2 * ref_freq ** dm_index)
            exp_arg_deriv = product / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = product_deriv * model.spectrum_per_component[freq_idx, :, current_component] / sc_time_freq
            term2 = product * deriv_first[freq_idx, :] / sc_time_freq
            term3 = -np.sqrt(2 / np.pi) * product_deriv * spectrum * np.exp(exp_arg) / burst_width / sc_time_freq
            term4 = -np.sqrt(2 / np.pi) * product * spectrum * np.exp(exp_arg) / burst_width / sc_time_freq * exp_arg_deriv
            deriv_mod_int[freq_idx, :, current_component] = term1 + term2 + term3 + term4

    return np.sum(deriv_mod_int, axis = 2)
