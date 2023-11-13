#! /bin/env/python

import matplotlib
matplotlib.rcParams["font.family"] = "times"
matplotlib.rcParams["font.size"] = 15
matplotlib.rcParams["xtick.labelsize"] = 12 
matplotlib.rcParams["ytick.labelsize"] = 12 

from copy import deepcopy
import matplotlib.pyplot as plt
import fitburst as fb
import numpy as np
import sys

# define dimensions of the data.
is_dedispersed = True
num_freq = 2 ** 8
num_time = 2 ** 7
freq_lo = 1200.
freq_hi = 1600.
time_lo = 0.
time_hi = 0.08

freqs = np.linspace(freq_lo, freq_hi, num = num_freq)  
times = np.linspace(time_lo, time_hi, num = num_time)  

# define physical parameters for a dispersed burst to simulate.
params = {                                                     
    "amplitude"            : [0., 0., 0.],
    "arrival_time"         : [0.03, 0.04, 0.05],
    "burst_width"          : [0.001, 0.002, 0.0005],
    "dm"                   : [349.5, 349.5, 349.5],
    "dm_index"             : [-2., -2., -2.],
    "ref_freq"             : [1500., 1400., 1300.],
    "scattering_index"     : [-4., -4., -4.],
    "scattering_timescale" : [0., 0., 0.],
    "spectral_index"       : [0., 0., 0.],
    "spectral_running"     : [-300., -300., -300.],
}  

num_components = len(params["dm"])

# define and/or extract parameters.
new_params = deepcopy(params)

if is_dedispersed:
    new_params["dm"] = [0.] * num_components

# define model object for CHIME/FRB data and load in parameter values.
model_obj = fb.analysis.model.SpectrumModeler(
            freqs,
            times,
            is_dedispersed = is_dedispersed,
            num_components = num_components,
            verbose = True,
        )

model_obj.update_parameters(new_params)

# now compute model and add noise.
model = model_obj.compute_model()
model += np.random.normal(0., 0.2, size = model.shape)

# plot.
plt.pcolormesh(times, freqs, model)
plt.xlabel("Time (s)")
plt.xlabel("Observing Frequency (MHz)")
plt.show()

# finally, save data into fitburst-generic format.
metadata = {
    "bad_chans" : [],
    "freqs_bin0" : freqs[0],
    "is_dedispersed" : is_dedispersed,
    "num_freq" : num_freq,
    "num_time" : num_time,
    "times_bin0" : 0.,
    "res_freq" : freqs[1] - freqs[0],
    "res_time" : times[1] - times[0]
}

np.savez(
    "simulated_data.npz",
    data_full = model,
    metadata = metadata,
    burst_parameters = params,    
)
