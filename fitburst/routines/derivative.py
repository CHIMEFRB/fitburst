"""
Routines for Partial Derivatives of the fitburst Model w.r.t. Fit Parameters

This module contains functions that return partial derivatives of the model 
defined by fitburt. Each derivative is only defined for fittable parameters, and 
so the fitter object will select which derivatives to compute based on 
fit parameters.
"""

from fitburst.backend import general
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

    burst_width = parameters["burst_width"][component]
    timediff = model.timediff_per_component[:, :, component]
    deriv_mod = timediff ** 2 * model.spectrum_per_component[:, :, component] / burst_width ** 3

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

    burst_width = parameters["burst_width"][component]
    timediff = model.timediff_per_component[:, :, component]
    deriv_mod = timediff * model.spectrum_per_component[:, :, component] / burst_width ** 2

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
 
    burst_width = parameters["burst_width"][component]
    ref_freq = parameters["ref_freq"][component]
    dm_const = general["constants"]["dispersion"]
    timediff = model.timediff_per_component[:, :, component]
    deriv_timediff_wrt_dm = -dm_const * (1 / model.freqs ** 2 - 1 / ref_freq ** 2)
    deriv_mod = -timediff * model.spectrum_per_component.sum(axis=2) / burst_width ** 2
    deriv_mod *= deriv_timediff_wrt_dm[:, None]

    return deriv_mod
