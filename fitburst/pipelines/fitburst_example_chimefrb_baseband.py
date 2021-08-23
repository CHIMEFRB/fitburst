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

# read in data.
data = DataReader("fitburst_65547659.npz", data_location="/data/frb-baseband/baseband_test")

# load data into memory and pre-process.
data.load_data()
data.good_freq = np.sum(data.data_weights, axis=1) // data.num_time

# get parameters.
initial_parameters = data.burst_parameters
current_parameters = deepcopy(initial_parameters)

if data.is_dedispersed:
    print("INFO: input data cube is already dedispersed!")
    print("INFO: setting 'dm' entry to 0, now considered a dm-offset parameter...")
    current_parameters["dm"][0] = 0.0

# now compute dedisperse matrix for data, given initial DM, and grab windowed data.
print("INFO: computing dedispersion-index matrix")
data.dedisperse(
    initial_parameters["dm"][0],
    current_parameters["arrival_time"][0],
    reference_freq=initial_parameters["ref_freq"][0],
    dm_offset=0.0
)

# get windowed data.
data_windowed, times_windowed = data.window_data(current_parameters["arrival_time"][0], window=0.002)

# now create model.
print("INFO: initializing model")
model = SpectrumModeler()
model.is_dedispersed = data.is_dedispersed
model.set_dimensions(data.num_freq, len(times_windowed))

# before instantiating model parameters, add a second (first-arriving) component.
# this step manually creates a two-component parameter dictionary that is needed
# for multi-component fitting.
num_components = 2
new_parameters = {}

for current_key, current_value in current_parameters.items():
    new_parameters[current_key] = current_parameters[current_key] * num_components

    # adjust arrival time of first/new burst.
    if current_key == "arrival_time":
        new_parameters[current_key][0] -= 0.0001

# add parameters to ensure all are set.
model.num_components = len(new_parameters["arrival_time"])
model.update_parameters(new_parameters)
model.update_parameters({"amplitude": [np.log10(np.mean(data_windowed))] * num_components})
model.update_parameters({"scattering_index": [-4.0] * num_components})
model.update_parameters({"scattering_timescale": [0.0] * num_components})
model.update_parameters({"spectral_index": [0.0] * num_components})
model.update_parameters({"spectral_running": [0.0] * num_components})

current_model = model.compute_model(times_windowed, data.freqs)

# now set up fitter and execute.
fitter = LSFitter(model)
fitter.fix_parameter(["dm", "dm_index", "scattering_timescale"])
fitter.weighted_fit = True
fitter.fit(times_windowed, data.freqs, data_windowed)

# extract best-fit data, create best-fit model and plot windowed data.
bestfit_parameters = fitter.load_fit_parameters_list(fitter.bestfit_results["parameters"])
bestfit_uncertainties = fitter.load_fit_parameters_list(fitter.bestfit_results["uncertainties"])
print(fitter.bestfit_results["solver"])
print("Best-fit parameters:", bestfit_parameters)
print("Best-fit uncertaintes:", bestfit_uncertainties)

# now compute best-fit model, residuals, and plot.
model.update_parameters(bestfit_parameters)
bestfit_model = model.compute_model(times_windowed, data.freqs)
bestfit_residuals = data_windowed - bestfit_model

ut.plotting.plot_summary_triptych(
    times_windowed, data.freqs, data_windowed, fitter.good_freq, model = bestfit_model,
    residuals = bestfit_residuals
)

# now re-dedisperse data and plot for comparison.
dm_offset = 0.

if "dm" in bestfit_parameters:
    dm_offset = bestfit_parameters["dm"][0]

print("INFO: computing dedispersion-index matrix")
data.dedisperse(
    initial_parameters["dm"][0],
    current_parameters["arrival_time"][0],
    reference_freq=initial_parameters["ref_freq"][0],
    dm_offset=dm_offset
)
data_windowed_new, times_windowed_new = data.window_data(current_parameters["arrival_time"][0], window=0.002)

plt.subplot(121)
plt.pcolormesh(manip.downsample_2d(data_windowed, 64, 1))
plt.title("Original")
plt.subplot(122)
plt.title("Re-Dedispersed")
plt.pcolormesh(manip.downsample_2d(data_windowed_new, 64, 1))
plt.tight_layout()
plt.savefig("spectra_comparisons.png", dpi=500, fmt="png")
