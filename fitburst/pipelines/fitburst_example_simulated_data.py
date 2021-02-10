#! /user/bin/env python

from fitburst.analysis.fitter import LSFitter
import fitburst.backend.chimefrb as chimefrb
import fitburst.analysis.model as mod
import chime_frb_constants as const
import numpy as np
import sys

# import and configure matplotlib.
import matplotlib.pyplot as plt

# read in data.
data = chimefrb.DataReader(37888771)
initial_parameters = data.get_parameters()
data.times = np.linspace(0., 0.08, num=100)
data.freqs = np.linspace(400., 800., num=1024)

# set up model.
model = mod.SpectrumModeler()
model.update_parameters(initial_parameters)
model.update_parameters({"arrival_time": [0.04]})
model.scattering_timescale = [0.0]
model.scattering_index = [-4.0]

# now set up fitter and create initial model.
fake_data = model.compute_model(data.times, data.freqs) 
fake_data += np.random.normal(size=fake_data.shape)
fitter = LSFitter(model)

# fix DM and scattering-timescale at their above values.
fitter.fix_parameter(["dm", "scattering_timescale"])

# now fit!
fitter.fit(data.times, data.freqs, fake_data)
