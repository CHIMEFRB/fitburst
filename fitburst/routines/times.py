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
