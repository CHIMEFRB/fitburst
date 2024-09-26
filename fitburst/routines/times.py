"""
Routines for Deriving Time-related Data
"""

from datetime import datetime, timedelta
import numpy as np
import pytz

def compute_arrival_times(parameters: dict, time0: float = 0.) -> list:
    """
    Computes arrival times on a specified timescale.

    Parameters
    ----------
    parameters : dict
        A dictionary that contains all burst-parameter data.

    time0 : float, optional
        A reference time against which the per-parameter arrival times are measured, 
        assumed to be in units of Unix time (i.e., seconds).

    Returns
    -------

    """

    

    # extract data.
    arrival_times = parameters["arrival_time"]
    arrival_times_converted = []
    dm = parameters["dm"][0]

    # construct datetime object.
    for current_timestamp in arrival_times:
        dt_arrival = timedelta(seconds=current_timestamp)
        dt_arrival += datetime.fromtimestamp(time0, tz=pytz.utc)
        arrival_times_converted += [dt_arrival.strftime("%Y-%m-%d %H:%M:%S.%f")]

    return arrival_times_converted

def compute_pulse_duration(parameters: dict, times: float) -> list:
    """
    Computes the duration of a pulse based on timestamp and pulse-model data.

    Parameters
    ----------
    parameters : dict
        A dictionary that contains all burst-parameter data.

    time0 : float
        an array of timestamps defining the time axis of the data subject to modeling.

    Returns
    -------
    pulse_durations : list
        a list containing two-element lists, each containing the indeces that denote the 'start' and 'end' 
        of duration for the corresponding burst component.
    """

    num_components = len(parameters["arrival_time"])
    sc_time = parameters["scattering_timescale"][0] # global parameter
    pulse_durations = []

    # loop over each component, compute start/end times and corresponding indeces.
    for current_component in range(num_components):
        arrival_time = parameters["arrival_time"][current_component]
        burst_width = parameters["burst_width"][current_component]
        start = arrival_time - burst_width
        end = arrival_time + burst_width

        # adjust for scatter-broadening, if relevant.
        if sc_time != 0.:
            end += sc_time

        # now determine bin indeces, and correct pre-burst bin if it is equal to the arrival bin.
        bins = np.digitize([start, end], times)
        bins[0] -= 1
        pulse_durations.append(bins.tolist())

    return pulse_durations
