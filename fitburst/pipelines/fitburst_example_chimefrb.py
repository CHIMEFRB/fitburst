#! /user/bin/env python

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

# now compute dedisperse matrix for data, given initial DM.
print("INFO: computing dedispersion-index matrix")
params = data.burst_parameters["fitburst"]["round_2"]
data.dedisperse(
    params["dm"][0],
    params["arrival_time"][0],
    reference_freq=params["reference_freq"]
)

# now create initial model.
model = mod.SpectrumModeler()
model.update_parameters(initial_parameters)
model.scattering_timescale = [0.0]
model.scattering_index = [-4.0]

# now obtain and plot windowed data.
plt.subplot(121)
data_windowed, times_windowed = data.window_data(params["arrival_time"][0])
plt.pcolormesh(times_windowed, data.freqs, data_windowed, cmap="inferno", vmin=0., vmax=0.3*np.max(data_windowed))

# also plot model.
plt.subplot(122)
current_model = model.compute_model(times_windowed, data.freqs)
plt.pcolormesh(times_windowed, data.freqs, current_model, cmap="inferno", vmin=0., vmax=np.max(current_model))
plt.savefig("test.png", dpi=500, fmt="png")
