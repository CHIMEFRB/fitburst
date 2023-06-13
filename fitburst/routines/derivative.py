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
    num_freq, num_time, num_component = model.timediff_per_component.shape
    deriv_mod = np.zeros((num_freq, num_time), dtype=float)

    # now loop over each model component and compute contribution to derivative.
    for component in range(num_component):
        amplitude = parameters["amplitude"][component]
        burst_width = parameters["burst_width"][component]
        ref_freq = parameters["ref_freq"][component]
        scattering_index = parameters["scattering_index"][component]
        scattering_timescale = parameters["scattering_timescale"][component]
        spectral_index = parameters["spectral_index"][component]
        spectral_running = parameters["spectral_running"][component]
        timediff = model.timediff_per_component[:, :, component]
        freq_ratio = model.freqs / ref_freq
        scat_times_freq = scattering_timescale * freq_ratio ** scattering_index

        for freq in range(num_freq):
            current_timediff = timediff[freq, :]

            if scat_times_freq[freq] < np.fabs(0.15 * burst_width):
                deriv_mod[freq, :] += (current_timediff ** 2 * model.spectrum_per_component[freq, :, component] 
                    / burst_width ** 3)

            else:

                # define argument of error and scattering timescales over frequency.
                log_freq = np.log(freq_ratio[freq])
                spectrum = 10 ** amplitude * freq_ratio[freq] ** (spectral_index + spectral_running * log_freq)
                current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width
                erf_arg = (current_timediff - burst_width ** 2  / scat_times_freq[freq]) / burst_width / np.sqrt(2)
                exp_erf = np.exp(-(erf_arg ** 2))
                exp_tau = np.exp((burst_width / scat_times_freq[freq]) ** 2 / 2)
                exp_tim = np.exp(-current_timediff / scat_times_freq[freq])

                # now compute derivative contribution from current component.
                term1 = burst_width * model.spectrum_per_component[freq, :, component] / scat_times_freq[freq] ** 2 
                term2 = -2 * erf_arg * exp_tau * exp_tim * exp_erf / np.sqrt(np.pi) / burst_width / scat_times_freq[freq] / 2
                term3 = -np.sqrt(2 / np.pi) * exp_tau * exp_tim * exp_erf / scat_times_freq[freq] ** 2 / 2
 
                deriv_mod[freq, :] += (term1 + (term2 + term3) * spectrum)

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
    for component in range(num_component):
        amplitude = parameters["amplitude"][component]
        burst_width = parameters["burst_width"][component]
        ref_freq = parameters["ref_freq"][component]
        scattering_index = parameters["scattering_index"][component]
        scattering_timescale = parameters["scattering_timescale"][component]
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
                current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width
                erf_arg = (current_timediff - burst_width ** 2  / scat_times_freq[freq]) / burst_width / np.sqrt(2)
                exp_erf = np.exp(-(erf_arg ** 2))
                exp_tau = np.exp((burst_width / scat_times_freq[freq]) ** 2 / 2)
                exp_tim = np.exp(-current_timediff / scat_times_freq[freq])

                # now compute derivative contribution from current component.
                term1 = model.spectrum_per_component[freq, :, component] / scat_times_freq[freq]
                term2 = -np.sqrt(2 / np.pi) * exp_erf * exp_tim * exp_tau / burst_width / scat_times_freq[freq] / 2

                deriv_mod[freq, :] += (term1 + term2 * spectrum)

    return deriv_mod

def deriv_model_wrt_dm(parameters: dict, model: float, component: int = 0) -> float:
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
    deriv_mod = np.zeros((num_freq, num_time), dtype=float)
    dm_const = 4149.377593360996

    # now loop over each model component and compute contribution to derivative.
    for component in range(num_component):
        amplitude = parameters["amplitude"][component]
        burst_width = parameters["burst_width"][component]
        dm_index = parameters["dm_index"][component]
        ref_freq = parameters["ref_freq"][component]
        scattering_index = parameters["scattering_index"][component]
        scattering_timescale = parameters["scattering_timescale"][component]
        spectral_index = parameters["spectral_index"][component]
        spectral_running = parameters["spectral_running"][component]
        timediff = model.timediff_per_component[:, :, component]
        freq_diff = model.freqs ** dm_index - ref_freq ** dm_index
        freq_ratio = model.freqs / ref_freq
        scat_times_freq = scattering_timescale * freq_ratio ** scattering_index

        for freq in range(num_freq):
            current_timediff = timediff[freq, :]

            if scat_times_freq[freq] < np.fabs(0.15 * burst_width):
                deriv_timediff_wrt_dm = -dm_const * freq_diff[freq]
                deriv_mod[freq, :] += (-current_timediff * model.spectrum_per_component[freq, :, component] 
                    / burst_width ** 2 * deriv_timediff_wrt_dm)

            else:

                # define argument of error and scattering timescales over frequency.
                log_freq = np.log(freq_ratio[freq])
                spectrum = 10 ** amplitude * freq_ratio[freq] ** (spectral_index + spectral_running * log_freq)
                current_timediff[current_timediff < -5 * burst_width] = -5 * burst_width
                erf_arg = (current_timediff - burst_width ** 2  / scat_times_freq[freq]) / burst_width / np.sqrt(2)
                exp_erf = np.exp(-(erf_arg ** 2))
                exp_tau = np.exp((burst_width / scat_times_freq[freq]) ** 2 / 2)
                exp_tim = np.exp(-current_timediff / scat_times_freq[freq])

                # now compute derivative contribution from current component.
                term1 = model.spectrum_per_component[freq, :, component] / scat_times_freq[freq]
                term2 = -np.sqrt(2 / np.pi) * exp_tau * exp_tim * exp_erf / burst_width / scat_times_freq[freq] / 2

                deriv_mod[freq, :] += dm_const * freq_diff[freq] * (term1 + term2 * spectrum)

    return deriv_mod

def deriv_model_wrt_dm_index(parameters: dict, model: float, component: int = 0) -> float:
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
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, : component]
    dm = parameters["dm"][0]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0]
    ref_freq = parameters["ref_freq"][component]
    
    print("model.freqs[:, None]: ", model.freqs.shape)
    print("current_model: ", current_model.shape)

    deriv_mod = -dm_const * dm * (np.log(model.freqs[:, None]) / model.freqs[:, None] ** dm_index - 
                np.log(ref_freq) / ref_freq ** dm_index) * current_model

    return deriv_mod 

def deriv_model_wrt_scattering_timescale(parameters: dict, model: float, component: int = 0) -> float:
    """
    Computes the derivative of the model with respect to the dm parameter.

    Parameters
    ----------
    parameters : dict
        A dictionary containing all values that instantiate the model

    model : object
        An instantiated version of the fitburst model object

    Returns
    -------
        The derivative of the model evaluated over time and frequency
    """
 
    # get dimensions and define empty model-derivative matrix.
    num_freq, num_time, num_component = model.timediff_per_component.shape
    deriv_mod = np.zeros((num_freq, num_time), dtype=float)

    # now loop over each model component and compute contribution to derivative.
    for component in range(num_component):
        amplitude = parameters["amplitude"][component]
        burst_width = parameters["burst_width"][component]
        ref_freq = parameters["ref_freq"][component]
        scattering_index = parameters["scattering_index"][component]
        scattering_timescale = parameters["scattering_timescale"][component]
        spectral_index = parameters["spectral_index"][component]
        spectral_running = parameters["spectral_running"][component]
        timediff = model.timediff_per_component[:, :, component]
        
        # define argument of error and scattering timescales over frequency.
        freq_ratio = model.freqs / ref_freq
        log_freq = np.log(freq_ratio)
        spectrum = 10 ** amplitude * freq_ratio ** (spectral_index + spectral_running * log_freq)
        timediff[timediff < -5 * burst_width] = -5 * burst_width
        scat_times_freq = scattering_timescale * freq_ratio ** scattering_index
        erf_arg = (timediff - burst_width ** 2  / scat_times_freq[:, None]) / burst_width / np.sqrt(2)
        product = np.exp((burst_width / scat_times_freq[:, None]) ** 2 / 2 - timediff / scat_times_freq[:, None]
            - erf_arg ** 2)

        # now compute derivative contribution from current component.
        term1 = model.spectrum_per_component[:, :, component] 
        term2 = (burst_width / scat_times_freq[:, None]) ** 2 * model.spectrum_per_component[:, :, component]
        term3 = -timediff / scat_times_freq[:, None] * model.spectrum_per_component[:, :, component]
        term4 = -np.sqrt(2 / np.pi) * product * burst_width / scat_times_freq[:, None] ** 2 / 2

        deriv_mod += -(term1 + term2 + term3 + term4 * spectrum[:, None]) / scattering_timescale

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
    deriv_mod = np.log(10) * deriv_model_wrt_dm(parameters, model, component)

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
    deriv_mod = np.log(10) * deriv_model_wrt_dm_index(parameters, model, component)

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
    deriv_mod = log_freq[:, None] * deriv_model_wrt_dm(parameters, model, component)

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
    deriv_mod = log_freq[:, None] * deriv_model_wrt_dm_index(parameters, model, component)

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
    deriv_mod = log_freq[:, None] * deriv_model_wrt_dm(parameters, model, component)

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
    deriv_mod = log_freq[:, None] * deriv_model_wrt_dm_index(parameters, model, component)

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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_mod = -3 * current_timediff ** 2 * current_model / burst_width ** 4
    deriv_mod += (current_timediff ** 4 * current_model / burst_width ** 6)

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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_mod = -2 * current_timediff * current_model / burst_width ** 3
    deriv_mod += (current_timediff ** 3 * current_model / burst_width ** 5)

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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]

    deriv_mod = -2 * current_timediff * dm_const * (model.freqs[:, None] ** dm_index - ref_freq ** dm_index) *\
                current_model / burst_width ** 3
    deriv_mod += (current_timediff ** 2 * deriv_model_wrt_dm(parameters, model, component) / burst_width ** 3)

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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    dm = parameters["dm"][0] # global parameter.
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    
    deriv_mod = -2 * current_timediff * dm_const * dm * (np.log(model.freqs[:, None]) * model.freqs[:, None] ** dm_index - \
                np.log(ref_freq) * ref_freq ** dm_index) * current_model / burst_width ** 3
    deriv_mod += (current_timediff ** 2 * deriv_model_wrt_dm_index(parameters, model, component) / burst_width ** 3)
    
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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    deriv_mod = current_timediff ** 2 * current_model / burst_width ** 4
    deriv_mod -= (current_model / burst_width ** 2)
    
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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]

    deriv_mod = current_timediff * deriv_model_wrt_dm(parameters, model, component) / burst_width ** 2
    deriv_mod -= (dm_const * (model.freqs[:, None] ** dm_index - ref_freq ** dm_index) * current_model / burst_width ** 2)

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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]

    deriv_mod = dm_const ** 2 * (model.freqs[:, None] ** dm_index - ref_freq ** dm_index) ** 2 * \
                current_timediff ** 2 * current_model / burst_width ** 4
    deriv_mod -= dm_const ** 2 * (model.freqs[:, None] ** dm_index - ref_freq ** dm_index) ** 2 * \
                 current_model / burst_width ** 2

    return deriv_mod

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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    dm = parameters["dm"][0] # global parameter.
    dm_const = 4149.377593360996
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]

    term1 = np.log(model.freqs) * model.freqs ** dm_index - np.log(ref_freq) * ref_freq ** dm_index
    term2 = np.log(model.freqs) ** 2 * model.freqs ** dm_index - np.log(ref_freq) ** 2 * ref_freq ** dm_index

    deriv_mod = (dm_const * dm * term1[:, None] * current_timediff / burst_width ** 2) ** 2 * current_model
    deriv_mod += dm_const * dm * current_timediff * term2[:, None] * current_model / burst_width ** 2
    deriv_mod -= current_model * (dm_const * dm * term1[:, None]) ** 2 * current_timediff / burst_width ** 2

    return deriv_mod
