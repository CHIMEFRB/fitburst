#! /user/bin/env python

from fitburst.analysis.fitter import LSFitter
import fitburst.backend.chimefrb as chimefrb
import fitburst.analysis.model as mod
import chime_frb_constants as const
import numpy as np
import sys

# import and configure matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# read in data.
data = chimefrb.DataReader(37888771)
initial_parameters = data.get_parameters()

# load data into memory and pre-process.
data.load_data(data.files)
data.preprocess_data(variance_weight=1/(const.L0_NUM_FRAMES_SAMPLE * 2))

# now compute dedisperse matrix for data, given initial DM, and grab windowed data.
print("INFO: computing dedispersion-index matrix")
params = data.burst_parameters["fitburst"]["round_2"]
data.dedisperse(
    params["dm"][0],
    params["arrival_time"][0],
    reference_freq=params["reference_freq"]
)

data_windowed, times_windowed = data.window_data(params["arrival_time"][0])

# now create initial model.
print("INFO: initializing model")
model = mod.SpectrumModeler()
model.set_dimensions(data.num_freq, len(times_windowed))
model.set_dedispersion_idx(data.dedispersion_idx)
model.update_parameters(initial_parameters)
model.dm_index = [-2.0]
model.scattering_timescale = [0.0]
model.scattering_index = [-4.0]

# now set up fitter and create initial model.
fitter = LSFitter(model)
fitter.fix_parameter(["dm_index", "scattering_timescale"])
results = fitter.fit(data.times, data.freqs, data_windowed)

# compute uncertainties.
jacT = results.jac.T.copy()
hessian = np.sum(jacT[:, None, :] * jacT[None, :, :], -1)
covariance = np.linalg.inv(hessian)
errs = np.diag(np.sqrt(covariance))
print(results.x)
print(errs)

# now obtain and plot windowed data.
plt.subplot(131)
plt.pcolormesh(times_windowed, data.freqs, data_windowed, cmap="inferno", vmin=0., vmax=0.3*np.max(data_windowed))

# also plot model.
bestfit_parameters = fitter.load_fit_parameters_list(results.x.tolist())
model.update_parameters(bestfit_parameters)
current_model = model.compute_model(data.times, data.freqs)

plt.subplot(132)
plt.pcolormesh(times_windowed, data.freqs, current_model, cmap="inferno", vmin=0., vmax=np.max(current_model))

plt.subplot(133)
resids = (data_windowed - current_model) * fitter.weights[:, None]
plt.pcolormesh(times_windowed, data.freqs, resids, cmap="inferno", vmin=np.min(resids), vmax=np.max(resids))

plt.savefig("test.png", dpi=500, fmt="png")
