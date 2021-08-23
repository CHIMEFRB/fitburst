#! /usr/bin/env python

# configure backend for matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fitburst.analysis.model import SpectrumModeler
from fitburst.backend.generic import DataReader
from fitburst.analysis.fitter import LSFitter
from copy import deepcopy
import fitburst.routines.manipulate as manip
import chime_frb_constants as const
import fitburst.utilities as ut
import numpy as np
import sys
import logging

from profile_modeling import get_signal
from baseband_analysis.utilities import get_profile
# Logging Config
LOGGING_CONFIG = {}
logging_format = "[%(asctime)s] %(process)d-%(levelname)s "
logging_format += "%(module)s::%(funcName)s():l%(lineno)d: "
logging_format += "%(message)s"
logging.basicConfig(format=logging_format, level=logging.INFO)
log = logging.getLogger()

def run_fitburst(fname, path):
    # read in data.
    data = DataReader(fname, data_location=path)
    # load data into memory and pre-process.
    data.load_data()
    data.good_freq = np.sum(data.data_weights, axis=1) // data.num_time
    # get parameters.
    initial_parameters = data.burst_parameters
    current_parameters = deepcopy(initial_parameters)

    # get windowed data.
    data_windowed, times_windowed = data.data_full, np.linspace(0,data.num_time,data.num_time) * data.res_time
    #data.window_data(np.mean(current_parameters["arrival_time"]), window=w)

    # now create model.
    log.info("Initializing model...")
    model = SpectrumModeler()
    model.is_dedispersed = data.is_dedispersed
    model.set_dimensions(data.num_freq, len(times_windowed))

    # before instantiating model parameters, add a second (first-arriving) component.
    # this step manually creates a two-component parameter dictionary that is needed
    # for multi-component fitting.
    num_components = len(current_parameters['arrival_time'])

    # add parameters to ensure all are set.
    model.num_components = len(current_parameters["arrival_time"])
    model.update_parameters(current_parameters)
    model.update_parameters({"amplitude": [np.log10(np.mean(data_windowed))] * num_components})
    model.update_parameters({"scattering_index": [-4.0] * num_components})
    model.update_parameters({"scattering_timescale": current_parameters["scattering_timescale"]})
    model.update_parameters({"spectral_index": current_parameters["spectral_index"]})
    model.update_parameters({"spectral_running": current_parameters["spectral_running"]})
    model.update_parameters({"ref_freq": current_parameters['ref_freq']})

    current_model = model.compute_model(times_windowed, data.freqs)

    # now set up fitter and execute.
    fitter = LSFitter(model)
    fitter.fix_parameter(['dm', "dm_index", "scattering_index"])
    fitter.weighted_fit = True
    fitter.fit(times_windowed, data.freqs, data_windowed)

    # extract best-fit data, create best-fit model and plot windowed data.
    bestfit_parameters = fitter.load_fit_parameters_list(fitter.bestfit_results["parameters"])
    try:
        bestfit_uncertainties = fitter.load_fit_parameters_list(fitter.bestfit_results["uncertainties"])
        print("Best-fit uncertaintes:", bestfit_uncertainties)
    except:
        pass
    print(fitter.bestfit_results["solver"])
    print("Best-fit parameters:", bestfit_parameters)
    

    # now compute best-fit model, residuals, and plot.
    model.update_parameters(bestfit_parameters)
    bestfit_model = model.compute_model(times_windowed, data.freqs)
    bestfit_residuals = data_windowed - bestfit_model

    ut.plotting.plot_summary_triptych(
       times_windowed, data.freqs, data_windowed, fitter.good_freq, num_std = 3, model = bestfit_model,
       residuals = bestfit_residuals, show = True
    )
    np.savez(path + 'RN1-2/' + fname, bestfit_parameters)
    return