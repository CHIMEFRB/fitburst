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
import fitburst.utilities as ut
import numpy as np
import argparse
import json
import sys
import os

infiles = (sys.argv)[1:]
fit_for_scattering = True
use_stored_results = True
parameters_fixed = ["dm_index", "scattering_index", "scattering_timescale"]
snr_threshold = 10.

# before proceeding, adjusted fixed-parameter list if necessary.
if fit_for_scattering:
    parameters_fixed.remove("scattering_timescale")

# now, loop over data files to read and perform fitting.
for current_file in infiles:
    elems = current_file.split("_")
    print(current_file)
    print(elems)
    filename_substring = "{0}_{1}".format(elems[1], elems[4])
    latest_solution_file = f"no_scattering/results_fitburst_J2108+45_{elems[4]}.json"

    if os.path.isfile(latest_solution_file) and use_stored_results:
        results = json.load(open(latest_solution_file, "r"))
    
    elif not use_stored_results:
        pass

    else:
        continue

    # now extract some numbers from previous fit.
    if results["fit_statistics"]["snr"] >= snr_threshold or not use_stored_results:

        data = DataReader(current_file)

        # load data into memory and pre-process.
        data.load_data()
        data.good_freq = np.sum(data.data_weights, axis=1) // data.num_time
        #data.preprocess_data(variance_range=[0., 0.8])

        # get parameters.
        initial_parameters = data.burst_parameters
        current_parameters = deepcopy(initial_parameters)

        if fit_for_scattering and use_stored_results:
            current_parameters = results["model_parameters"]
            current_parameters["amplitude"] = [-3.]
            current_parameters["arrival_time"] = [0.55]
            current_parameters["burst_width"] = [0.008]
            current_parameters["dm"] = [0.0]
            current_parameters["scattering_timescale"] = [0.18]
            current_parameters["spectral_index"] = [-1.0]
            current_parameters["spectral_running"] = [1.0]

        else:
            current_parameters["arrival_time"] = [0.5]
            current_parameters["burst_width"] = [0.05]
            current_parameters["scattering_timescale"] = [0.0]
            current_parameters["spectral_index"] = [-1.0]
            current_parameters["spectral_running"] = [1.0]

        if data.is_dedispersed and not use_stored_results:
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

        data_windowed = data.data_full
        times_windowed = data.times
        #plt.pcolormesh(data.times, data.freqs, data.data_full)
        #plt.savefig("test.png")

        # now create model.
        print("INFO: initializing model")
        model = SpectrumModeler()
        model.dm0 = initial_parameters["dm"][0]
        model.is_dedispersed = data.is_dedispersed
        model.is_folded = True
        model.set_dimensions(data.num_freq, len(times_windowed))
        model.update_parameters(current_parameters)
        current_model = model.compute_model(times_windowed, data.freqs)

        # now set up fitter and create initial model.
        fitter = LSFitter(model)
        fitter.fix_parameter(parameters_fixed)
        fitter.weighted_fit = True
        fitter.fit(times_windowed, data.freqs, data_windowed)

        # extract best-fit data, create best-fit model and plot windowed data.
        if fitter.success:
            bestfit_parameters = fitter.fit_statistics["bestfit_parameters"]
            bestfit_uncertainties = fitter.fit_statistics["bestfit_uncertainties"]
            model.update_parameters(bestfit_parameters)
            bestfit_model = model.compute_model(times_windowed, data.freqs)
            bestfit_residuals = data_windowed - bestfit_model

            ut.plotting.plot_summary_triptych(
                times_windowed, data.freqs, data_windowed, fitter.good_freq, model = bestfit_model,
                residuals = bestfit_residuals, output_name = f"summary_{filename_substring}.png",
            )

            with open(f"results_fitburst_{filename_substring}.json", "w") as out:
                json.dump(
                    {
                        "initial_dm": initial_parameters["dm"][0],
                        "initial_time": data.times_bin0,
                        "model_parameters": model.get_parameters_dict(),
                        "fit_statistics": fitter.fit_statistics,
                    },
                    out,
                    indent=4
                )
