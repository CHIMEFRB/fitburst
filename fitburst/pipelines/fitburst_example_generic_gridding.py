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

parser = argparse.ArgumentParser(description=
    "A Python3 script that uses fitburst API to read, preprocess, window, and fit CHIME/FRB data " +
    "against a model of the dynamic spectrum."
)

parser.add_argument(
    "file", action="store", nargs="+", type=str,
    help="Data file containing spectrum and metada in 'generic' format."
)

parser.add_argument(
    "-n", action="store", dest="num_grid", default=20, type=int,
    help="Number of grid points along each dimension"
)

parser.add_argument(
    "--solution", action="store", dest="solution", nargs="+", type=str,
    help="Data file containing spectrum and metada in 'generic' format."
)

### grab CLI inputs from argparse.
args = parser.parse_args()
infiles = args.file
num_grid = args.num_grid
insolutions = args.solution
fit_for_scattering = True
use_stored_results = True
parameters_fixed = ["dm_index", "scattering_index", "scattering_timescale"]
snr_threshold = 10.

# before proceeding, adjusted fixed-parameter list if necessary.
if fit_for_scattering:
    parameters_fixed.remove("scattering_timescale")

# now, loop over data files to read and perform fitting.
for current_file, current_solution in zip(infiles, insolutions):
    current_file_base = current_file.split("/")
    elems = current_file_base[-1].split("_")
    print(current_file)
    print(elems)
    filename_substring = "{0}_{1}".format(elems[1], elems[4])

    if os.path.isfile(current_solution) and use_stored_results:
        results = json.load(open(current_solution, "r"))
    
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
        print("INFO: now executing least-squares fitting...")
        fitter.fit(times_windowed, data.freqs, data_windowed)

        # extract best-fit data, create best-fit model and plot windowed data.
        if fitter.success:
            bestfit_parameters = fitter.fit_statistics["bestfit_parameters"]
            bestfit_uncertainties = fitter.fit_statistics["bestfit_uncertainties"]
            print(bestfit_parameters)
            model.update_parameters(bestfit_parameters)

            # now define gridding bounds.
            min_dm = bestfit_parameters["dm"][0] - 10 * bestfit_uncertainties["dm"][0]
            max_dm = bestfit_parameters["dm"][0] + 10 * bestfit_uncertainties["dm"][0]
            min_st = bestfit_parameters["scattering_timescale"][0] - 10 * bestfit_uncertainties["scattering_timescale"][0]
            max_st = bestfit_parameters["scattering_timescale"][0] + 10 * bestfit_uncertainties["scattering_timescale"][0]

            if min_st < 0.:
                min_st = 0.

            # next, define grid arrays.
            array_dm = np.linspace(min_dm, max_dm, num=num_grid)
            array_st = np.linspace(min_st, max_st, num=num_grid)

            # now, loop over dimensions and determine fit.
            current_parameters_grid = deepcopy(fitter.fit_statistics["bestfit_parameters"])
            matrix_chisq = np.zeros((num_grid, num_grid))
            del fitter

            for ii in range(num_grid):
                for jj in range(num_grid):

                    # update model with adjusted values for gridded parameters.
                    current_parameters_grid["dm"] = [array_dm[ii]]
                    current_parameters_grid["scattering_timescale"] = [array_st[jj]]
                    model.update_parameters(current_parameters_grid)

                    # now define fitter for gridded parameters.
                    fitter = LSFitter(model)
                    fitter.fix_parameter(parameters_fixed + ["dm", "scattering_timescale"])
                    fitter.weighted_fit = True
                    fitter.fit(times_windowed, data.freqs, data_windowed)
                    matrix_chisq[ii, jj] = fitter.fit_statistics["chisq_final"]

            # now compute PDF map and plot.
            print(matrix_chisq)
            pdf_chisq = 0.5 * np.exp(-0.5 * (matrix_chisq - np.min(matrix_chisq)))
            plt.pcolormesh(array_dm, array_st, pdf_chisq, cmap="Blues")
            plt.xlabel(r"Dispersion Measure (pc cm$^{-3}$)")
            plt.ylabel(r"Scattering Timescale @ 400 MHz (ms)")
            plt.savefig("pdf_map.png", dpi=500, fmt="png")
