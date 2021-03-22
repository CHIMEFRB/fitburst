#! /usr/bin/env python

from fitburst.backend.generic import DataReader
from copy import deepcopy
import chime_frb_constants as const
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
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
    current_parameters["dm"][0],
    current_parameters["arrival_time"][0],
    reference_freq=initial_parameters["ref_freq"][0]
)

# get windowed data.
data_windowed, times_windowed = data.window_data(current_parameters["arrival_time"][0], window=0.04)

plt.pcolormesh(times_windowed, data.freqs, data_windowed)
plt.savefig("test.png", fmt="png")
