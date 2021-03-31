#! /user/bin/env python

from fitburst.analysis.fitter import LSFitter
import fitburst.backend.chimefrb as chimefrb
import fitburst.analysis.model as mod
import chime_frb_constants as const
import fitburst.utilities as ut
import fitburst.routines as rt
import numpy as np
import sys

# import and configure matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# read in data.
# test "simple" burst: 37888771
# test multi-component burst: 156410110
data = chimefrb.DataReader(156410110)
initial_parameters = data.get_parameters()

# load data into memory and pre-process.
data.load_data(data.files)
data.preprocess_data(variance_weight=1./(const.L0_NUM_FRAMES_SAMPLE * 2))

# now compute dedisperse matrix for data, given initial DM, and grab windowed data.
print("INFO: computing dedispersion-index matrix")
params = data.burst_parameters["fitburst"]["round_2"]
data.dedisperse(
    params["dm"][0],
    params["arrival_time"][0],
    reference_freq=params["reference_freq"][0]
)

data_windowed, times_windowed = data.window_data(params["arrival_time"][0], window=0.08)

# now create initial model.
print("INFO: initializing model")
model = mod.SpectrumModeler()
model.is_dedispersed = False
model.set_dimensions(data.num_freq, len(times_windowed))
model.set_dedispersion_idx(data.dedispersion_idx)
model.update_parameters(initial_parameters)
model.dm_index = [-2.0]
model.scattering_timescale = [0.0]
model.scattering_index = [-4.0]

# now set up fitter and create initial model.
fitter = LSFitter(model)
fitter.fix_parameter(["dm_index", "scattering_timescale"])
fitter.fit(data.times, data.freqs, data_windowed)

# now obtain and plot windowed data.
bestfit_parameters = fitter.load_fit_parameters_list(fitter.bestfit_results["parameters"])
model.update_parameters(bestfit_parameters)
bestfit_model = model.compute_model(data.times, data.freqs)
bestfit_residuals = data_windowed - bestfit_model

ut.plotting.plot_summary_triptych(
    data.times, data.freqs, data_windowed, data.good_freq, model = bestfit_model, 
    residuals = bestfit_residuals
)
