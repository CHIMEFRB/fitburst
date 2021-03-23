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
data = DataReader("test_data.npz")

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
    reference_freq=initial_parameters["reference_freq"][0],
    dm_offset=0.0
)

# get windowed data.
data_windowed, times_windowed = data.window_data(current_parameters["arrival_time"][0], window=0.02)

# now create model.
print("INFO: initializing model")
model = SpectrumModeler()
model.dm0 = initial_parameters["dm"][0]
model.is_dedispersed = data.is_dedispersed
model.set_dimensions(data.num_freq, len(times_windowed))
model.update_parameters(current_parameters)
current_model = model.compute_model(times_windowed, data.freqs)

# now set up fitter and create initial model.
fitter = LSFitter(model)
fitter.fix_parameter(["dm_index", "scattering_timescale"])
fitter.weighted_fit = False
fitter.fit(times_windowed, data.freqs, data_windowed)

# extract best-fit data, create best-fit model and plot windowed data.
bestfit_parameters = fitter.load_fit_parameters_list(fitter.bestfit_results["parameters"])
bestfit_uncertainties = fitter.load_fit_parameters_list(fitter.bestfit_results["uncertainties"])
print("Best-fit parameters:", bestfit_parameters)
print("Best-fit uncertaintes:", bestfit_uncertainties)
model.update_parameters(bestfit_parameters)
bestfit_model = model.compute_model(times_windowed, data.freqs)
bestfit_residuals = data_windowed - bestfit_model

ut.plotting.plot_summary_triptych(
    times_windowed, data.freqs, data_windowed, fitter.good_freq, model = bestfit_model,
    residuals = bestfit_residuals
)

# now re-dedisperse data and plot for comparison.
print("INFO: computing dedispersion-index matrix")
data.dedisperse(
    initial_parameters["dm"][0],
    current_parameters["arrival_time"][0],
    reference_freq=initial_parameters["reference_freq"][0],
    dm_offset=bestfit_parameters["dm"][0]
)
data_windowed_new, times_windowed_new = data.window_data(current_parameters["arrival_time"][0], window=0.02)

plt.subplot(121)
plt.pcolormesh(manip.downsample_2d(data_windowed, factor=64))
plt.title("Original")
plt.subplot(122)
plt.title("Re-Dedispersed")
plt.pcolormesh(manip.downsample_2d(data_windowed_new, factor=64))
plt.tight_layout()
plt.savefig("spectra_comparisons.png", dpi=500, fmt="png")
