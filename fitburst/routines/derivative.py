"""
Routines for Partial Derivatives of the fitburst Model w.r.t. Fit Parameters

This module contains functions that return partial derivatives of the model
defined by fitburt. Each derivative is only defined for fittable parameters, and
so the fitter object will select which derivatives to compute based on
fit parameters.
"""

from ..backend import general
import scipy.special as ss
import numpy as np

def argument_erf(burst_width: float, time_diff: float, scattering_timescale: float) -> float:
    """
    Computes the argument of the error function in the scatter-broadened fitburst model.

    Parameters
    ----------
    burst_width : float
        the temporal width of the burst component

    time_diff : float
        the dispersion-delayed timeseries relative to the arrival time

    scattering_timescale : float
        the scattering timescale at the frequency corresponding to 'time_diff'

    Returns
    -------
    arg_erf : float
        the argument of the error function
    """

    # compute and return the argument of the error function.
    arg_erf = (time_diff - burst_width ** 2 / scattering_timescale) / burst_width / np.sqrt(2)

    return arg_erf

def argument_exp(burst_width: float, time_diff: float, scattering_timescale: float) -> float:
    """
    Computes the argument of the exponential term that is common to all derivatives
    of the scatter-broadened fitburst model.

    Parameters
    ----------

    burst_width : float
        the temporal width of the burst component

    time_diff : float
        the dispersion-delayed timeseries relative to the arrival time

    scattering_timescale : float
        the scattering timescale at the frequency corresponding to 'time_diff'

    Returns
    -------
    arg_exp : float
         the argument of the exponential term
    """

    # compute the argument to the error function.
    arg_erf = argument_erf(burst_width, time_diff, scattering_timescale)

    # compute and return the argument of the exponential term.
    arg_exp = 0.5 * (burst_width / scattering_timescale) ** 2 - time_diff / scattering_timescale - arg_erf ** 2

    return arg_exp

def amplitude_pbf(freq: float, parameters: dict, component: int):
    """
    Computes the amplitude of the pulse-broadening-function (PBF).

    Parameters
    ----------
    freq : float
        value of the electromagnetic frequency at which to evaluate dispersed timeseries

    parameters : dict
        A dictionary containing all values that instantiate the model

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
    amp : float
        Amplitude of the PBF model
    """
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter

    amp = (freq / ref_freq) ** -sc_index

    if general["flags"]["normalize_pbf"]:

        sc_time = parameters["scattering_timescale"][0] # global parameter
        burst_width = parameters["burst_width"][component]

        amp *= np.sqrt(np.pi / 2.0) * burst_width / sc_time

    return amp

def deriv_amplitude_pbf(name: str, freq: float, parameters: dict, component: int):
    """
    Computes the derivative of the PBF amplitude with respect to either the
    burst-width or scattering-timescale parameters.

    Parameters
    ----------
    name : str
        name of the fit parameter for which to compute the partial derivative

    freq : float
        value of the electromagnetic frequency at which to evaluate dispersed timeseries

    parameters : dict
        A dictionary containing all values that instantiate the model

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
    deriv_partial : float
        partial derivative of the PBF amplitude with respect to the 'name' fit parameter
    """

    burst_width = parameters["burst_width"][component]
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter
    sc_time = parameters["scattering_timescale"][0] # global parameter

    deriv_partial = 0.0
    if general["flags"]["normalize_pbf"]:

        sc_time_freq = sc_time * (freq / ref_freq) ** sc_index

        if name == "burst_width":
            deriv_partial = np.sqrt(np.pi / 2) / sc_time_freq

        elif name == "scattering_index":
            deriv_partial = -np.sqrt(np.pi / 2) * np.log(freq / ref_freq) * burst_width / sc_time_freq

        elif name == "scattering_timescale":
            deriv_partial = -np.sqrt(np.pi / 2) * burst_width / (sc_time_freq * sc_time)

    elif name == "scattering_index":
        deriv_partial = - np.log(freq / ref_freq) *  (freq / ref_freq) ** -sc_index

    return deriv_partial

def deriv2_amplitude_pbf(name1: str, name2: str, freq: float, parameters: dict, component: int):
    """
    Computes the mixed partial derivative of the PBF amplitude with
    respect to two fitburst parameters that define a scatter-broadened model.

    Parameters
    ----------
    name1 : str
        name of the fit parameter for which to compute the partial derivative

    name2 : str
        name of the fit parameter for which to compute the partial derivative

    freq : float
        value of the electromagnetic frequency at which to evaluate dispersed timeseries

    parameters : dict
        A dictionary containing all values that instantiate the model

    component : int, optional
        An index of the burst component for which to evaluate the derivative

    Returns
    -------
    deriv_partial : float
        partial derivative of the PBF amplitude with respect to the 'name' fit parameter
    """

    burst_width = parameters["burst_width"][component]
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter
    sc_time = parameters["scattering_timescale"][0] # global parameter

    if general["flags"]["normalize_pbf"]:

        sc_time_freq = sc_time * (freq / ref_freq) ** sc_index

        if (name1 == "burst_width" and name2 == "scattering_index") or (name2 == "burst_width" and name1 == "scattering_index") :
            deriv_partial = -np.sqrt(np.pi / 2) * np.log(freq / ref_freq) / sc_time_freq

        elif (name1 == "burst_width" and name2 == "scattering_timescale") or (name2 == "burst_width" and name1 == "scattering_timescale"):
            deriv_partial = -np.sqrt(np.pi / 2) / (sc_time_freq * sc_time)

        elif (name1 == "scattering_index" and name2 == "scattering_index"):
            deriv_partial = np.sqrt(np.pi / 2) * np.log(freq / ref_freq) ** 2 * burst_width / sc_time_freq

        elif (name1 == "scattering_index" and name2 == "scattering_timescale") or (name2 == "scattering_index" and name1 == "scattering_timescale"):
            deriv_partial = np.sqrt(np.pi / 2) * np.log(freq / ref_freq) * burst_width / (sc_time_freq * sc_time)

    elif name1 == "scattering_index" and name2 == "scattering_index":
        deriv_partial = np.log(freq / ref_freq) ** 2 *  (freq / ref_freq) ** -sc_index

    else:
        deriv_partial = 0.0

    return deriv_partial

def deriv_time_dm(name: str, freq: float, ref_freq: float, dm: float, dm_index: float):
    """
    Computes the derivative of the dispersed timeseries with respect to either the DM
    of DM-index parameters.

    Parameters
    ----------
    name : str
        name of the fit parameter for which to compute the partial derivative

    freq : float
        value of the electromagnetic frequency at which to evaluate dispersed timeseries

    ref_freq : float
        value of the electromagnetic frequency at which to reference dispersion

    dm : float
        value of dispersion measure

    dm_index : float
        value of exponent for frequency dependence of dispersion relation

    Returns
    -------
    deriv_partial : float
         partial derivative of the dispersed timeseries with respect to the 'name' fit parameter
    """

    deriv_partial = 0.
    dm_const = general["constants"]["dispersion"]

    if name == "dm":
        diff_freq = freq ** dm_index - ref_freq ** dm_index
        deriv_partial = -dm_const * diff_freq

    elif name == "dm_index":
        diff_freq = np.log(freq) * freq ** dm_index - np.log(ref_freq) * ref_freq ** dm_index
        deriv_partial = -dm_const * dm * diff_freq

    return deriv_partial

def deriv_argument_erf(name: str, freq: float, time_diff: float, parameters: dict,
    component: int = 0) -> float:
    """
    Computes the derivative of the error-function argument.

    Parameters
    ----------
    name : str
        name of the fit parameter for which to compute the partial derivative

    burst_width : float
        the temporal width of the burst component

    time_diff : float
        the dispersion-delayed timeseries relative to the arrival time

    sc_time : float
        the scattering timescale at the frequency corresponding to 'time_diff'

    sc_time_ref : float
        the scattering timescale at the frequency corresponding to 'time_diff'

    Returns
    -------
    arg_exp : float
         the argument of the exponential term
    """

    # get parameters.
    width = parameters["burst_width"][component]
    dm = parameters["dm"][0]
    dm_index = parameters["dm_index"][0]
    ref_freq = parameters["ref_freq"][component]
    sc_time = parameters["scattering_timescale"][0]
    sc_index = parameters["scattering_index"][0]
    sc_time_freq = sc_time * (freq / ref_freq) ** sc_index

    # compute derivative with respect to the appropriate parameter.
    deriv_first = 0.

    if name == "arrival_time":
        deriv_first = -1 / width / np.sqrt(2)

    elif name == "burst_width":
        deriv_first = -(time_diff / width ** 2 + 1 / sc_time_freq) / np.sqrt(2)

    elif name == "dm" or name == "dm_index":
        deriv_first = deriv_time_dm(name, freq, ref_freq, dm, dm_index) / width / np.sqrt(2)

    elif name == "scattering_timescale":
        deriv_first = width / sc_time / sc_time_freq / np.sqrt(2)

    elif name == "scattering_index":
        deriv_first = np.log(freq / ref_freq) * width / sc_time_freq / np.sqrt(2)

    return deriv_first

def deriv_argument_exp(name: str, freq: float, time_diff: float, parameters: dict,
    component: int = 0) -> float:
    """
    Computes the argument of the exponential term that is common to all derivatives
    of the scatter-broadened fitburst model.

    Parameters
    ----------
    name : str
        name of the fit parameter for which to compute the partial derivative

    burst_width : float
        the temporal width of the burst component

    time_diff : float
        the dispersion-delayed timeseries relative to the arrival time

    sc_time : float
        the scattering timescale at the frequency corresponding to 'time_diff'

    sc_time_ref : float
        the scattering timescale at the frequency corresponding to 'time_diff'

    Returns
    -------
    arg_exp : float
         the argument of the exponential term
    """

    # get parameters.
    width = parameters["burst_width"][component]
    dm = parameters["dm"][0]
    dm_index = parameters["dm_index"][0]
    ref_freq = parameters["ref_freq"][component]
    sc_time = parameters["scattering_timescale"][0]
    sc_index = parameters["scattering_index"][0]
    sc_time_freq = sc_time * (freq / ref_freq) ** sc_index
    arg_erf = argument_erf(width, time_diff, sc_time_freq)
    deriv_arg_erf = deriv_argument_erf(name, freq, time_diff, parameters, component)

    # compute derivative with respect to the appropriate parameter.
    deriv_first = 0.

    if name == "arrival_time":
        deriv_first = 1 / sc_time_freq

    elif name == "burst_width":
        deriv_first = width / sc_time_freq ** 2

    elif name == "dm" or name == "dm_index":
        deriv_first = -deriv_time_dm(name, freq, ref_freq, dm, dm_index) / sc_time_freq

    elif name == "scattering_timescale":
        deriv_first = -(width / sc_time_freq) ** 2 / sc_time + time_diff / sc_time_freq / sc_time

    elif name == "scattering_index":
        log_freq = np.log(freq / ref_freq)
        deriv_first = log_freq * (-(width / sc_time_freq) ** 2 + time_diff / sc_time_freq)

    # now apply subtraction that is common to all derivatives.
    deriv_first -= 2 * arg_erf * deriv_arg_erf

    return deriv_first

def deriv2_argument_erf(name1: str, name2: str, freq: float, time_diff: float, parameters: dict,
    component: int = 0) -> float:
    """
    Computes the mixed-partial derivative of the error-function argument with
    respect to one or two fitburst parameters that define a scatter-broadened model.

    Parameters
    ----------
    name1 : str
        name of the fit parameter for which to compute the partial derivative

    name2 : str
        name of the fit parameter for which to compute the partial derivative

    burst_width : float
        the temporal width of the burst component

    time_diff : float
        the dispersion-delayed timeseries relative to the arrival time

    sc_time : float
        the scattering timescale at the frequency corresponding to 'time_diff'

    sc_time_ref : float
        the scattering timescale at the frequency corresponding to 'time_diff'

    Returns
    -------
    arg_exp : float
         the argument of the exponential term
    """

    # get parameters.
    width = parameters["burst_width"][component]
    dm = parameters["dm"][0]
    dm_index = parameters["dm_index"][0]
    ref_freq = parameters["ref_freq"][component]
    sc_time = parameters["scattering_timescale"][0]
    sc_index = parameters["scattering_index"][0]
    sc_time_freq = sc_time * (freq / ref_freq) ** sc_index
    deriv_mixed = 0.

    if name1 == "arrival_time" and name2 == "arrival_time":
        pass # mixed partial evaluates to 0.

    elif name1 == "arrival_time" and name2 == "burst_width":
        deriv_mixed = 1 / width ** 2 / np.sqrt(2)

    elif name1 == "arrival_time" and name2 == "scattering_timescale":
        pass # mixed partial evaluates to 0.

    elif name1 == "arrival_time" and name2 == "dm":
        pass # mixed partial evaluates to 0.

    elif name1 == "burst_width" and name2 == "burst_width":
        deriv_mixed = np.sqrt(2) * time_diff / width ** 3

    elif (name1 == "burst_width" and name2 == "dm") or (name1 == "burst_width" and name2 == "dm_index"):
        deriv_mixed = -deriv_time_dm(name2, freq, ref_freq, dm, dm_index) / width ** 2 / np.sqrt(2)

    elif name1 == "burst_width" and name2 == "scattering_index":
        log_freq = np.log(freq / ref_freq)
        deriv_mixed = log_freq / np.sqrt(2) / sc_time_freq

    elif name1 == "burst_width" and name2 == "scattering_timescale":
        deriv_mixed = 1 / sc_time / sc_time_freq / np.sqrt(2)

    elif name1 == "dm" and name2 == "dm":
        pass # mixed partial evaluates to 0.

    elif name1 == "dm" and name2 == "scattering_timescale":
        pass # mixed partial evaluates to 0.

    elif name1 == "scattering_timescale" and name2 == "scattering_timescale":
        deriv_mixed = -np.sqrt(2) * width / sc_time_freq / sc_time ** 2

    else:
        sys.exit(f"ERROR: unrecognized deriv2_argument_erf pair: {name1}, {name2}")

    return deriv_mixed

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
    burst_width = parameters["burst_width"][component]
    current_amplitude = model.amplitude_per_component[:, :, component]
    current_model = model.spectrum_per_component[:, :, component]
    num_freq, num_time, num_component = model.timediff_per_component.shape
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    timediff = model.timediff_per_component[:, :, component]
    sc_times_freq = sc_time * (model.freqs / ref_freq) ** sc_index
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros((num_freq, num_time), dtype=float)

    for freq in range(num_freq):
        current_timediff = timediff[freq, :]

        if sc_times_freq[freq] == 0.0:
            deriv_mod[freq, :] += (current_timediff ** 2 * current_model[freq, :] / burst_width ** 3)

        else:
            # define argument of error and scattering timescales over frequency.
            arg_exp = argument_exp(burst_width, current_timediff, sc_times_freq[freq])
            arg_erf = argument_erf(burst_width, current_timediff, sc_times_freq[freq])
            arg_exp_model = 0.5 * (burst_width / sc_times_freq[freq]) ** 2 - current_timediff / sc_times_freq[freq]
            deriv_amp_pbf = deriv_amplitude_pbf(
                "burst_width", model.freqs[freq], parameters, component
            )
            deriv_arg_erf = deriv_argument_erf(
                "burst_width", model.freqs[freq], current_timediff, parameters, component
            )

            # now compute derivative contribution from current component.
            term1 = current_model[freq, :] * deriv_amp_pbf / amp_pbf[freq]
            term2 = (burst_width / sc_times_freq[freq] ** 2) * current_model[freq, :]
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * 2 / np.sqrt(np.pi)

            deriv_mod[freq, :] += term1 + term2 + term3

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
    burst_width = parameters["burst_width"][component]
    current_amplitude = model.amplitude_per_component[:, :, component]
    current_model = model.spectrum_per_component[:, :, component]
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    timediff = model.timediff_per_component[:, :, component]
    sc_times_freq = sc_time * (model.freqs / ref_freq) ** sc_index
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    for freq in range(num_freq):
        current_timediff = timediff[freq, :]

        if sc_times_freq[freq] == 0.0:
            deriv_mod[freq, :] += current_timediff * current_model[freq, :] / burst_width ** 2

        else:
            # define argument of error and scattering timescales over frequency.a
            arg_exp = argument_exp(burst_width, current_timediff, sc_times_freq[freq])
            arg_erf = argument_erf(burst_width, current_timediff, sc_times_freq[freq])
            arg_exp_model = 0.5 * (burst_width / sc_times_freq[freq]) ** 2 - current_timediff / sc_times_freq[freq]
            deriv_amp_pbf = deriv_amplitude_pbf(
                "arrival_time", model.freqs[freq], parameters, component
            )
            deriv_arg_erf = deriv_argument_erf(
                "arrival_time", model.freqs[freq], current_timediff, parameters, component
            )

            # now compute derivative contribution from current component.
            term1 = current_model[freq, :] * deriv_amp_pbf / amp_pbf[freq]
            term2 = current_model[freq, :] / sc_times_freq[freq]
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * 2 / np.sqrt(np.pi)

            deriv_mod[freq, :] = term1 + term2 + term3

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
    dm = parameters["dm"][0]
    dm_index = parameters["dm_index"][0]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]

    # now loop over each model component and compute contribution to derivative.
    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_model = model.spectrum_per_component[:, :, current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        ref_freq = parameters["ref_freq"][current_component]
        sc_times_freq = sc_time * (model.freqs / ref_freq) ** sc_index
        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)
        timediff = model.timediff_per_component[:, :, current_component]
        deriv_timediff_wrt_dm = deriv_time_dm("dm", model.freqs, ref_freq, dm, dm_index)

        for freq in range(num_freq):
            current_timediff = timediff[freq, :]

            if sc_times_freq[freq] == 0.0:
                deriv_mod_int[freq, :, current_component] = (-current_timediff *
                    current_model[freq, :] / burst_width ** 2 * deriv_timediff_wrt_dm[freq])

            else:

                # define argument of error and scattering timescales over frequency.
                arg_exp = argument_exp(burst_width, current_timediff, sc_times_freq[freq])
                arg_erf = argument_erf(burst_width, current_timediff, sc_times_freq[freq])
                arg_exp_model = 0.5 * (burst_width / sc_times_freq[freq]) ** 2 - current_timediff / sc_times_freq[freq]
                deriv_amp_pbf = deriv_amplitude_pbf(
                    "dm", model.freqs[freq], parameters, component
                )
                deriv_arg_erf = deriv_argument_erf(
                    "dm", model.freqs[freq], current_timediff, parameters, current_component
                )

                # now compute derivative contribution from current component.
                term1 = current_model[freq, :] * deriv_amp_pbf / amp_pbf[freq]
                term2 = -deriv_timediff_wrt_dm[freq] * current_model[freq, :] / sc_times_freq[freq]
                term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * 2 / np.sqrt(np.pi)
                deriv_mod_int[freq, :, current_component] = term1 + term2 + term3

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
    num_freq, num_time, num_component = model.timediff_per_component.shape
    deriv_mod_int = np.zeros(model.spectrum_per_component.shape)
    dm = parameters["dm"][0]
    dm_index = parameters["dm_index"][0]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        current_model = model.spectrum_per_component[:, :, current_component]
        timediff = model.timediff_per_component[:, : , current_component]
        ref_freq = parameters["ref_freq"][current_component]
        sc_times_freq = sc_time * (model.freqs / ref_freq) ** sc_index
        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)
        deriv_timediff_wrt_dm = deriv_time_dm("dm_index", model.freqs, ref_freq, dm, dm_index)

        for freq in range (num_freq):
            current_timediff = timediff[freq, :]

            if sc_times_freq[freq] == 0.0:
                deriv_mod_int[freq, :, current_component] += (-current_timediff *
                    current_model[freq, :] / burst_width ** 2 * deriv_timediff_wrt_dm[freq])

            else:

                # define argument of error and scattering timescales over frequency.
                arg_exp = argument_exp(burst_width, current_timediff, sc_times_freq[freq])
                arg_erf = argument_erf(burst_width, current_timediff, sc_times_freq[freq])
                arg_exp_model = 0.5 * (burst_width / sc_times_freq[freq]) ** 2 - current_timediff / sc_times_freq[freq]
                deriv_amp_pbf = deriv_amplitude_pbf(
                    "dm_index", model.freqs[freq], parameters, component
                )
                deriv_arg_erf = deriv_argument_erf(
                    "dm_index", model.freqs[freq], current_timediff, parameters, current_component
                )

                # now compute derivative contribution from current component.
                term1 = current_model[freq, :] * deriv_amp_pbf / amp_pbf[freq]
                term2 = -deriv_timediff_wrt_dm[freq] * model.spectrum_per_component[freq, :, current_component]
                term2 /= sc_times_freq[freq]
                term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * 2 / np.sqrt(np.pi)
                deriv_mod_int[freq, :, current_component] = term1 + term2 + term3

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
        burst_width = parameters["burst_width"][current_component]
        current_model = model.spectrum_per_component[:, :, current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        ref_freq = parameters["ref_freq"][current_component]
        sc_index = parameters["scattering_index"][0]
        sc_time = parameters["scattering_timescale"][0]
        sc_times_freq = sc_time * (model.freqs / ref_freq) ** sc_index
        amp_pbf = amplitude_pbf(model.freqs[:, None], parameters, current_component)

        current_timediff = model.timediff_per_component[:, :, current_component]

        # define argument of error and scattering timescales over frequency.
        arg_exp = argument_exp(burst_width, current_timediff, sc_times_freq[:, None])
        arg_erf = argument_erf(burst_width, current_timediff, sc_times_freq[:, None])
        arg_exp_model = 0.5 * (burst_width / sc_times_freq[:, None]) ** 2 - current_timediff / \
                        sc_times_freq[:, None]
        deriv_amp_pbf = deriv_amplitude_pbf(
            "scattering_timescale", model.freqs[:, None], parameters, component
        )
        deriv_arg_erf = deriv_argument_erf(
            "scattering_timescale", model.freqs[:, None], current_timediff, parameters, current_component
        )

        # now compute derivative contribution from current component.
        term1 = current_model * deriv_amp_pbf / amp_pbf
        term2 = (-(burst_width / sc_times_freq[:, None]) ** 2 + current_timediff / sc_times_freq[:, None]) * \
                current_model[:, :] / sc_time
        term3 = current_amplitude * amp_pbf * np.exp(arg_exp) * deriv_arg_erf * 2 / np.sqrt(np.pi)

        deriv_mod_int[:, :, current_component] = term1 + term2 + term3

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
        current_model = model.spectrum_per_component[:, :, current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        ref_freq = parameters["ref_freq"][current_component]
        freq_ratio = model.freqs / ref_freq
        log_freq = np.log(freq_ratio)
        sc_index = parameters["scattering_index"][0]
        sc_time = parameters["scattering_timescale"][0]
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, :, current_component]
        sc_times_freq = sc_time * freq_ratio ** sc_index
        amp_pbf = amplitude_pbf(model.freqs[:, None], parameters, current_component)

        # define argument of error and scattering timescales over frequency.
        arg_exp = argument_exp(burst_width, current_timediff, sc_times_freq[:, None])
        arg_erf = argument_erf(burst_width, current_timediff, sc_times_freq[:, None])
        arg_exp_model = 0.5 * (burst_width / sc_times_freq[:, None]) ** 2 - current_timediff / sc_times_freq[:, None]
        deriv_amp_pbf = deriv_amplitude_pbf(
            "scattering_index", model.freqs[:, None], parameters, component
        )
        deriv_arg_erf = deriv_argument_erf(
            "scattering_index", model.freqs[:, None], current_timediff, parameters, current_component
        )

        # now compute derivative contribution from current component.
        term1 = current_model * deriv_amp_pbf / amp_pbf
        term2 = -log_freq[:, None] * current_model * ((burst_width / sc_times_freq[:, None]) ** 2 -
                                                      current_timediff / sc_times_freq[:, None])
        term3 = current_amplitude * amp_pbf * np.exp(arg_exp) * deriv_arg_erf * 2 / np.sqrt(np.pi)

        deriv_mod_int[:, :, current_component] = term1 + term2 + term3

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
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_burst_width(parameters, model, component)
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros(current_model.shape)
    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq in range(current_model.shape[0]):
        sc_time_freq = sc_time * (model.freqs[freq] / ref_freq) ** sc_index

        # if scattering is not resolvable, then assume Gaussian temporal profile.
        if sc_time_freq == 0.0:
            deriv_mod[freq, :] = -3 * current_timediff[freq, :] ** 2 * \
                                     current_model[freq, :] / burst_width ** 4
            deriv_mod[freq, :] += (current_timediff[freq, :] ** 4 * \
                                      current_model[freq, :] / burst_width ** 6)

        else:
            arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
            deriv_arg_erf = deriv_argument_erf(
                "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv_arg_exp = deriv_argument_exp(
                "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv2_arg_erf = deriv2_argument_erf(
                "burst_width", "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
            )

            # now define terms that contribute to mixed derivative.
            term1 = current_model[freq] / sc_time_freq ** 2
            term2 = burst_width * deriv_first[freq, :] / sc_time_freq ** 2
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
            term4 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)
            deriv_mod[freq, :] = term1 + term2 + term3 + term4

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
    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_arrival_time(parameters, model, component)
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros(current_model.shape)

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq in range(current_model.shape[0]):
        sc_time_freq = sc_time * (model.freqs[freq] / ref_freq) ** sc_index

        # if scattering is not resolvable, then assume Gaussian temporal profile.
        if sc_time_freq == 0.0:
            deriv_mod[freq, :] = -2 * current_timediff[freq, :] * current_model[freq, :] / burst_width ** 3
            deriv_mod[freq, :] += (current_timediff[freq, :] ** 3 * current_model[freq, :] / burst_width ** 5)

        else:
            arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
            deriv_arg_erf = deriv_argument_erf(
                "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv_arg_exp = deriv_argument_exp(
                "arrival_time", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv2_arg_erf = deriv2_argument_erf(
                "arrival_time", "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
            )

            # now define terms that contriubte to mixed-partial derivative.
            term1 = burst_width * deriv_first[freq, :] / sc_time_freq ** 2
            term2 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)
            deriv_mod[freq, :] = term1 + term2 + term3

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
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_scattering_timescale(parameters, model, component, add_all = False)
    burst_width = parameters["burst_width"][component]
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    spectral_index = parameters["spectral_index"][component]
    spectral_running = parameters["spectral_running"][component]
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros(current_model.shape)

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq in range(current_model.shape[0]):
        sc_time_freq = sc_time * (model.freqs[freq] / ref_freq) ** sc_index
        arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
        deriv_arg_erf = deriv_argument_erf(
            "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
        )
        deriv_arg_exp = deriv_argument_exp(
            "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, component
        )
        deriv2_arg_erf = deriv2_argument_erf(
            "burst_width", "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, component
        )

        # now define terms that contriubte to mixed-partial derivative.
        term1 = -2 * burst_width * current_model[freq, :] / sc_time / sc_time_freq ** 2
        term2 = burst_width * deriv_first[freq, :] / sc_time_freq ** 2
        term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
        term4 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)
        deriv_mod[freq, :] = term1 + term2 + term3 + term4

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
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_scattering_index(parameters, model, component, add_all = False)
    burst_width = parameters["burst_width"][component]
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros(current_model.shape)

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq] / ref_freq
        log_freq = np.log(freq_ratio)
        sc_time_freq = sc_time * freq_ratio ** sc_index
        arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
        deriv_arg_erf = deriv_argument_erf(
            "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
        )
        deriv_arg_exp = deriv_argument_exp(
            "scattering_index", model.freqs[freq], current_timediff[freq, :], parameters, component
        )
        deriv2_arg_erf = deriv2_argument_erf(
            "burst_width", "scattering_index", model.freqs[freq], current_timediff[freq, :], parameters, component
        )

        # now define terms that contriubte to mixed-partial derivative.
        term1 = -2 * log_freq * burst_width * current_model[freq, :] / sc_time_freq ** 2
        term2 = burst_width * deriv_first[freq, :] / sc_time_freq ** 2
        term3 = -current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * log_freq * 2 / np.sqrt(np.pi)
        term4 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
        term5 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)

        deriv_mod[freq, :] = term1 + term2 + term3 + term4 + term5

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
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_dm(parameters, model, component, add_all = False)
    dm = parameters["dm"][0] # global parameter.
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_first = deriv_model_wrt_dm(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape)

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq in range(current_model.shape[0]):
        sc_time_freq = sc_time * (model.freqs[freq] / ref_freq) ** sc_index

        # if scattering is not resolvable, then assume Gaussian temporal profile.
        if sc_time_freq == 0.0:
            deriv_time_dm_freq = deriv_time_dm("dm", model.freqs[freq], ref_freq, dm, dm_index)
            deriv_mod[freq, :] += 2 * current_timediff[freq, :] * deriv_time_dm_freq * \
                                      current_model[freq, :] / burst_width ** 3
            deriv_mod[freq, :] += (current_timediff[freq, :] ** 2 * deriv_first[freq, :] / burst_width ** 3)

        else:
            arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
            deriv_arg_erf = deriv_argument_erf(
                "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv_arg_exp = deriv_argument_exp(
                "dm", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv2_arg_erf = deriv2_argument_erf(
                "burst_width", "dm", model.freqs[freq], current_timediff[freq, :], parameters, component
            )

            # now define terms that contriubte to mixed-partial derivative.
            term1 = burst_width * deriv_first[freq, :] / sc_time_freq ** 2
            term2 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)
            deriv_mod[freq, :] = term1 + term2 + term3

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
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_dm_index(parameters, model, component, add_all = False)
    dm = parameters["dm"][0] # global parameter.
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    dm_index = parameters["dm_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    sc_index = parameters["scattering_index"][0] # global parameter.
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros(current_model.shape)

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq in range(current_model.shape[0]):
        sc_time_freq = sc_time * (model.freqs[freq] / ref_freq) ** sc_index

        # if scattering is not resolvable, then assume Gaussian temporal profile.
        if sc_time_freq == 0.0:
            deriv_time_dm_freq = deriv_time_dm("dm_index", model.freqs[freq], ref_freq, dm, dm_index)
            deriv_mod[freq, :] += -2 * current_timediff[freq, :] * deriv_time_dm_freq * current_model[freq, :] / burst_width ** 3
            deriv_mod[freq, :] += (current_timediff[freq, :] ** 2 * deriv_first[freq, :] / burst_width ** 3)

        else:
            # adjust time-difference values to make them friendly for error function.
            arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
            deriv_arg_erf = deriv_argument_erf(
                "burst_width", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv_arg_exp = deriv_argument_exp(
                "dm_index", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv2_arg_erf = deriv2_argument_erf(
                "burst_width", "dm_index", model.freqs[freq], current_timediff[freq, :], parameters, component
            )

            # now define terms that contriubte to mixed-partial derivative.
            term1 = burst_width * deriv_first[freq, :] / sc_time_freq ** 2
            term2 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)

            deriv_mod[freq, :] = term1 + term2 + term3

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
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_arrival_time(parameters, model, component)
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros((num_freq, num_time), dtype=float)

    for freq in range(num_freq):
        sc_time_freq = sc_time * (model.freqs[freq] / ref_freq) ** sc_index

        if sc_time_freq == 0.0:
            deriv_mod[freq, :] = current_timediff[freq, :] ** 2 * current_model[freq, :] / burst_width ** 4
            deriv_mod[freq, :] -= (current_model[freq, :] / burst_width ** 2)

        else:
            arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
            deriv_arg_erf = deriv_argument_erf(
                "arrival_time", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv_arg_exp = deriv_argument_exp(
                "arrival_time", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv2_arg_erf = deriv2_argument_erf(
                "arrival_time", "arrival_time", model.freqs[freq], current_timediff[freq, :], parameters, component
            )

            term1 = deriv_first[freq, :] / sc_time_freq
            term2 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)
            deriv_mod[freq, :] = term1 + term2 + term3

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
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_dm(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape, dtype=float)
    dm = parameters["dm"][0] # global parameter.
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros(current_model.shape, dtype=float)

    # now loop over each frequency and compute mixed-derivative array per channel.
    deriv_timediff_wrt_dm = deriv_time_dm("dm", model.freqs, ref_freq, dm, dm_index)

    for freq in range(current_model.shape[0]):
        sc_time_freq = sc_time * (model.freqs[freq] / ref_freq) ** sc_index

        if sc_time_freq == 0.0:
            deriv_mod[freq, :] = current_timediff[freq, :] * deriv_first[freq, :] / burst_width ** 2
            deriv_mod[freq, :] += deriv_timediff_wrt_dm[freq] * current_model[freq, :] / burst_width ** 2

        else:
            arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
            deriv_arg_erf = deriv_argument_erf(
                "arrival_time", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv_arg_exp = deriv_argument_exp(
                "dm", model.freqs[freq], current_timediff[freq, :], parameters, component
            )
            deriv2_arg_erf = deriv2_argument_erf(
                "arrival_time", "dm", model.freqs[freq], current_timediff[freq, :], parameters, component
            )

            # now define terms that contriubte to mixed-partial derivative.
            term1 = deriv_first[freq, :] / sc_time_freq
            term2 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)
            deriv_mod[freq, :] = term1 + term2 + term3

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
    dm_const = general["constants"]["dispersion"]
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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_scattering_timescale(parameters, model, component, add_all = False)
    dm = parameters["dm"][0] # global parameter.
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    dm_index = parameters["dm_index"][0] # global parameter.
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    sc_index = parameters["scattering_index"][0] # global parameter.
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    deriv_mod = np.zeros(current_model.shape)

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq in range(current_model.shape[0]):
        sc_time_freq = sc_time * (model.freqs[freq] / ref_freq) ** sc_index
        arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
        deriv_arg_erf = deriv_argument_erf(
            "arrival_time", model.freqs[freq], current_timediff[freq, :], parameters, component
        )
        deriv_arg_exp = deriv_argument_exp(
            "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, component
        )
        deriv2_arg_erf = deriv2_argument_erf(
            "arrival_time", "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, component
        )

        # now define terms that contriubte to mixed-partial derivative.
        term1 = -model.spectrum_per_component[freq, :, component] / sc_time / sc_time_freq
        term2 = deriv_first[freq, :] / sc_time_freq
        term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
        term4 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)
        deriv_mod[freq, :] = term1 + term2 + term3 + term4

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

    burst_width = parameters["burst_width"][component]
    current_model = model.spectrum_per_component[:, :, component]
    current_timediff = model.timediff_per_component[:, :, component]
    current_amplitude = model.amplitude_per_component[:, :, component]
    deriv_first = deriv_model_wrt_scattering_index(parameters, model, component, add_all = False)
    deriv_mod = np.zeros(current_model.shape)
    dm = parameters["dm"][0] # global parameter.
    dm_index = parameters["dm_index"][0] # global parameter.
    ref_freq = parameters["ref_freq"][component]
    sc_time = parameters["scattering_timescale"][0] # global parameter.
    sc_index = parameters["scattering_index"][0] # global parameter.
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    # now loop over each frequency and compute mixed-derivative array per channel.
    for freq in range(current_model.shape[0]):
        freq_ratio = model.freqs[freq] / ref_freq
        sc_time_freq = sc_time * freq_ratio ** sc_index

        amp_pbf_deriv = deriv_amplitude_pbf("scattering_index", model.freqs[freq], parameters, component)

        erf_arg = (current_timediff[freq, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
        erf_arg_deriv1 = -1.0 / (burst_width * np.sqrt(2))
        erf_arg_deriv2 = np.log(freq_ratio) * burst_width / (sc_time_freq * np.sqrt(2))

        exp_arg = burst_width ** 2 / 2 / sc_time_freq ** 2 - current_timediff[freq, :] / sc_time_freq - erf_arg ** 2
        exp_arg_deriv = -burst_width ** 2 * np.log(freq_ratio) / sc_time_freq ** 2 + \
                        current_timediff[freq, :] * np.log(freq_ratio) / sc_time_freq - \
                        2 * erf_arg * erf_arg_deriv2

        # now define terms that contriubte to mixed-partial derivative.
        term1 = -model.spectrum_per_component[freq, :, component] * np.log(freq_ratio) / sc_time_freq
        term2 = deriv_first[freq, :] / sc_time_freq
        term3 = 2.0 * current_amplitude[freq, :] * np.exp(exp_arg) * amp_pbf_deriv * erf_arg_deriv1 /  np.sqrt(np.pi)
        term4 = 2.0 * current_amplitude[freq, :] * np.exp(exp_arg) * amp_pbf[freq] * erf_arg_deriv1 * exp_arg_deriv / np.sqrt(np.pi)

        deriv_mod[freq, :] = term1 + term2 + term3 + term4

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
    dm_const = general["constants"]["dispersion"]
    dm = parameters["dm"][0] # global parameter.
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
    dm = parameters["dm"][0] # global parameter.
    dm_index = parameters["dm_index"][0] # global parameter.
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]

    for current_component in range(model.spectrum_per_component.shape[2]):
        burst_width = parameters["burst_width"][current_component]
        current_model = model.spectrum_per_component[:, :, current_component]
        current_timediff = model.timediff_per_component[:, :, current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_dm(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)

        # now loop over each frequency and compute mixed-derivative array per channel.
        deriv_timediff_wrt_dm = deriv_time_dm("dm", model.freqs, ref_freq, dm, dm_index)

        for freq in range(current_model.shape[0]):
            freq_ratio = model.freqs[freq] / ref_freq
            freq_diff = model.freqs[freq] ** dm_index - ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index

            if sc_time_freq == 0.0:
                deriv_mod_int[freq, :, current_component] = (deriv_timediff_wrt_dm[freq] * current_timediff[freq, :]) ** 2 * \
                    current_model[freq, :] / burst_width ** 4
                deriv_mod_int[freq, :, current_component] -= (deriv_timediff_wrt_dm[freq] / burst_width) ** 2 * current_model[freq, :]

            else:
                arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
                deriv_arg_erf = deriv_argument_erf(
                    "dm", model.freqs[freq], current_timediff[freq, :], parameters, current_component
                )
                deriv_arg_exp = deriv_argument_exp(
                    "dm", model.freqs[freq], current_timediff[freq, :], parameters, current_component
                )
                deriv2_arg_erf = deriv2_argument_erf(
                    "dm", "dm", model.freqs[freq], current_timediff[freq, :], parameters, current_component
                )

                term1 = -deriv_timediff_wrt_dm[freq] * deriv_first[freq, :] / sc_time_freq
                term2 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
                term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)
                deriv_mod_int[freq, :, current_component] = term1 + term2 + term3

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
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_scattering_timescale(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)

        for freq in range(num_freq):
            freq_ratio = model.freqs[freq] / ref_freq
            sc_time_freq = sc_time * freq_ratio ** sc_index
            arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
            deriv_arg_erf = deriv_argument_erf(
                "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, current_component
            )
            deriv_arg_exp = deriv_argument_exp(
                "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, current_component
            )
            deriv2_arg_erf = deriv2_argument_erf(
                "scattering_timescale", "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, current_component
            )

            # now compute derivative terms here.
            term0 = (current_timediff[freq, :] / sc_time_freq - (burst_width / sc_time_freq) ** 2) / sc_time
            term0_deriv = -(current_timediff[freq, :] / sc_time_freq - (burst_width / sc_time_freq) ** 2) / sc_time ** 2 + \
                           (-current_timediff[freq, :] / sc_time_freq + 2 * (burst_width / sc_time_freq) ** 2) / sc_time ** 2
            term1 = term0_deriv * model.spectrum_per_component[freq, :, current_component]
            term2 = term0 * deriv_first[freq, :]
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
            term4 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)

            deriv_mod_int[freq, :, current_component] = term1 + term2 + term3 + term4

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
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_model2 = deriv_model_wrt_scattering_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]

        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)

        for freq in range(num_freq):
            freq_ratio = model.freqs[freq] / ref_freq
            sc_time_freq = sc_time * freq_ratio ** sc_index

            erf_arg = (current_timediff[freq, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv1 = burst_width / (np.sqrt(2.0) * sc_time * sc_time_freq)
            erf_arg_deriv2 = burst_width * np.log(freq_ratio) / sc_time_freq / np.sqrt(2)
            erf_arg_deriv12 = -np.log(freq_ratio) * erf_arg_deriv1

            amp_pbf_deriv2 = deriv_amplitude_pbf(
                "scattering_index", model.freqs[freq], parameters, current_component
            )

            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv2 = (-(burst_width / sc_time_freq) ** 2 * np.log(freq_ratio) +
                              current_timediff[freq, :] * np.log(freq_ratio) / sc_time_freq - 2 * erf_arg * erf_arg_deriv2)

            term0 = (-burst_width ** 2 / sc_time_freq ** 2 / sc_time + current_timediff[freq, :] / sc_time / sc_time_freq)
            term0_deriv2 = (2 * (burst_width / sc_time_freq) ** 2 / sc_time * np.log(freq_ratio) -
                            current_timediff[freq, :] * np.log(freq_ratio) / sc_time_freq / sc_time)

            term1 = term0_deriv2 * model.spectrum_per_component[freq, :, current_component]
            term2 = term0 * deriv_model2[freq, :]
            term3 = 2.0 * current_amplitude[freq, :] * np.exp(exp_arg) * amp_pbf_deriv2 * erf_arg_deriv1 /  np.sqrt(np.pi)
            term4 = 2.0 * current_amplitude[freq, :] * np.exp(exp_arg) * amp_pbf[freq] * erf_arg_deriv1 * exp_arg_deriv2 / np.sqrt(np.pi)
            term5 = 2.0 * current_amplitude[freq, :] * np.exp(exp_arg) * amp_pbf[freq] * erf_arg_deriv12 / np.sqrt(np.pi)

            deriv_mod_int[freq, :, current_component] = term1 + term2 + term3 + term4 + term5

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
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_model = model.spectrum_per_component[:, :, current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_scattering_timescale(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        deriv_timediff_wrt_dm = deriv_time_dm("dm", model.freqs, ref_freq, dm, dm_index)
        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)

        # TODO: replace lines below.
        for freq in range(num_freq):
            freq_ratio = model.freqs[freq] / ref_freq
            sc_time_freq = sc_time * freq_ratio ** sc_index
            arg_exp = argument_exp(burst_width, current_timediff[freq, :], sc_time_freq)
            deriv_arg_erf = deriv_argument_erf(
                "dm", model.freqs[freq], current_timediff[freq, :], parameters, current_component
            )
            deriv_arg_exp = deriv_argument_exp(
                "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, current_component
            )
            deriv2_arg_erf = deriv2_argument_erf(
                "dm", "scattering_timescale", model.freqs[freq], current_timediff[freq, :], parameters, current_component
            )

            term1 = -deriv_timediff_wrt_dm[freq] * deriv_first[freq, :] / sc_time_freq
            term2 = deriv_timediff_wrt_dm[freq] * current_model[freq, :] / sc_time / sc_time_freq
            term3 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv2_arg_erf * 2 / np.sqrt(np.pi)
            term4 = current_amplitude[freq, :] * amp_pbf[freq] * np.exp(arg_exp) * deriv_arg_erf * deriv_arg_exp * 2 / np.sqrt(np.pi)

            deriv_mod_int[freq, :, current_component] = term1 + term2 + term3 + term4

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
    dm_const = general["constants"]["dispersion"]
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_dm_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)

        for freq in range(num_freq):
            freq_ratio = model.freqs[freq] / ref_freq
            freq_diff = np.log(model.freqs[freq]) * model.freqs[freq] ** dm_index - \
                        np.log(ref_freq) * ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = (-1 / sc_time - burst_width ** 2 / sc_time_freq ** 2 / sc_time + current_timediff[freq, :] / \
                    sc_time / sc_time_freq)
            term0_deriv = -dm_const * dm * freq_diff / sc_time_freq / sc_time
            erf_arg = (current_timediff[freq, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -dm_const * dm * freq_diff / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = dm_const * dm * freq_diff / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = term0_deriv * model.spectrum_per_component[freq, :, current_component]
            term2 = term0 * deriv_first[freq, :]
            term3 = np.sqrt(2 / np.pi) * current_amplitude[freq, :] * amp_pbf[freq] * burst_width * np.exp(exp_arg) * exp_arg_deriv / sc_time_freq / sc_time

            deriv_mod_int[freq, :, current_component] = term1 + term2 + term3

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
    dm_const = general["constants"]["dispersion"]
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_scattering_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]

        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)

        for freq in range(num_freq):
            freq_ratio = model.freqs[freq] / ref_freq
            freq_diff = np.log(model.freqs[freq]) * model.freqs[freq] ** dm_index - \
                        np.log(ref_freq) * ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index

            amp_pbf_deriv = deriv_amplitude_pbf("scattering_index", model.freqs[freq], parameters, current_component)
            amp_pbf_deriv2 = deriv2_amplitude_pbf("scattering_index", "scattering_index", model.freqs[freq], parameters, current_component)

            erf_arg = (current_timediff[freq, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = burst_width * np.log(freq_ratio) / sc_time_freq / np.sqrt(2)
            erf_arg_deriv2 = -burst_width * np.log(freq_ratio) ** 2 / sc_time_freq / np.sqrt(2)

            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = -np.log(freq_ratio) * (burst_width / sc_time_freq) ** 2 + \
                            current_timediff[freq, :] * np.log(freq_ratio) / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term0 = amp_pbf_deriv / amp_pbf[freq] - (burst_width ** 2 / sc_time_freq ** 2 - current_timediff[freq, :] / sc_time_freq) * np.log(freq_ratio)
            term0_deriv = (amp_pbf_deriv2 / amp_pbf[freq] - (amp_pbf_deriv / amp_pbf[freq]) ** 2 -
                           np.log(freq_ratio) * (-2 * np.log(freq_ratio) * (burst_width / sc_time_freq) ** 2 +
                                                  current_timediff[freq, :] * np.log(freq_ratio) / sc_time_freq))

            term1 = term0_deriv * model.spectrum_per_component[freq, :, current_component]
            term2 = term0 * deriv_first[freq, :]
            term3 = 2.0 * current_amplitude[freq, :] * np.exp(exp_arg) * amp_pbf_deriv * erf_arg_deriv /  np.sqrt(np.pi)
            term4 = 2.0 * current_amplitude[freq, :] * np.exp(exp_arg) * amp_pbf[freq] * erf_arg_deriv * exp_arg_deriv / np.sqrt(np.pi)
            term5 = 2.0 * current_amplitude[freq, :] * np.exp(exp_arg) * amp_pbf[freq] * erf_arg_deriv2 / np.sqrt(np.pi)

            deriv_mod_int[freq, :, current_component] = term1 + term2 + term3 + term4 + term5

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
    dm_const = general["constants"]["dispersion"]
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_dm(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)

        for freq in range(num_freq):
            freq_ratio = model.freqs[freq] / ref_freq
            freq_diff = model.freqs[freq] ** dm_index - ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = -(1 + burst_width ** 2 / sc_time_freq ** 2 - current_timediff[freq, :] / sc_time_freq) * np.log(freq_ratio)
            term0_deriv = -np.log(freq_ratio) * dm_const * freq_diff / sc_time_freq
            erf_arg = (current_timediff[freq, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -dm_const * freq_diff / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = dm_const * freq_diff / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = term0_deriv * model.spectrum_per_component[freq, :, current_component]
            term2 = term0 * deriv_first[freq, :]
            term3 = np.sqrt(2 / np.pi) * current_amplitude[freq, :] * amp_pbf[freq] * burst_width * np.exp(exp_arg) * np.log(freq_ratio) * exp_arg_deriv / sc_time_freq
            deriv_mod_int[freq, :, current_component] = term1 + term2 + term3

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
    dm_const = general["constants"]["dispersion"]
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_dm_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]
        amp_pbf = amplitude_pbf(model.freqs, parameters, current_component)

        for freq in range(num_freq):
            freq_ratio = model.freqs[freq] / ref_freq
            freq_diff = np.log(model.freqs[freq]) * model.freqs[freq] ** dm_index - \
                        np.log(ref_freq) * ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            term0 = -(1 + burst_width ** 2 / sc_time_freq ** 2 - current_timediff[freq, :] / sc_time_freq) * np.log(freq_ratio)
            term0_deriv = -np.log(freq_ratio) * dm_const * dm * freq_diff / sc_time_freq
            erf_arg = (current_timediff[freq, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -dm_const * dm * freq_diff / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq, :] / sc_time_freq - erf_arg ** 2
            exp_arg_deriv = dm_const * dm * freq_diff / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = term0_deriv * model.spectrum_per_component[freq, :, current_component]
            term2 = term0 * deriv_first[freq, :]
            term3 = np.sqrt(2 / np.pi) * current_amplitude[freq, :] * amp_pbf[freq] * burst_width * np.exp(exp_arg) * np.log(freq_ratio) * exp_arg_deriv / sc_time_freq
            deriv_mod_int[freq, :, current_component] = term1 + term2 + term3

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
    dm_const = general["constants"]["dispersion"]
    dm_index = parameters["dm_index"][0]
    num_freq, num_time, num_component = model.spectrum_per_component.shape
    sc_index = parameters["scattering_index"][0]
    sc_time = parameters["scattering_timescale"][0]
    deriv_mod_int = np.zeros((num_freq, num_time, num_component), dtype=float)
    amp_pbf = amplitude_pbf(model.freqs, parameters, component)

    for current_component in range(num_component):
        burst_width = parameters["burst_width"][current_component]
        current_timediff = model.timediff_per_component[:, : , current_component]
        current_amplitude = model.amplitude_per_component[:, :, current_component]
        deriv_first = deriv_model_wrt_dm_index(parameters, model, current_component, add_all = False)
        ref_freq = parameters["ref_freq"][current_component]

        for freq in range(num_freq):
            freq_ratio = model.freqs[freq] / ref_freq
            freq_diff = np.log(model.freqs[freq]) * model.freqs[freq] ** dm_index - \
                        np.log(ref_freq) * ref_freq ** dm_index
            sc_time_freq = sc_time * freq_ratio ** sc_index
            erf_arg = (current_timediff[freq, :] / burst_width - burst_width / sc_time_freq) / np.sqrt(2)
            erf_arg_deriv = -dm_const * dm * freq_diff / burst_width / np.sqrt(2)
            exp_arg = (burst_width / sc_time_freq) ** 2 / 2 - current_timediff[freq, :] / sc_time_freq - erf_arg ** 2
            product = dm_const * dm * (np.log(model.freqs[freq]) * model.freqs[freq] ** dm_index -\
                      np.log(ref_freq) * ref_freq ** dm_index)
            product_deriv = dm_const * dm * (np.log(model.freqs[freq]) ** 2 * model.freqs[freq] ** dm_index -\
                            np.log(ref_freq) ** 2 * ref_freq ** dm_index)
            exp_arg_deriv = product / sc_time_freq - 2 * erf_arg * erf_arg_deriv

            term1 = product_deriv * model.spectrum_per_component[freq, :, current_component] / sc_time_freq
            term2 = product * deriv_first[freq, :] / sc_time_freq
            term3 = -np.sqrt(2 / np.pi) * product_deriv * current_amplitude[freq, :] * amp_pbf[freq] * np.exp(exp_arg) / burst_width / sc_time_freq
            term4 = -np.sqrt(2 / np.pi) * product * current_amplitude[freq, :] * amp_pbf[freq] * np.exp(exp_arg) / burst_width / sc_time_freq * exp_arg_deriv
            deriv_mod_int[freq, :, current_component] = term1 + term2 + term3 + term4

    return np.sum(deriv_mod_int, axis = 2)
