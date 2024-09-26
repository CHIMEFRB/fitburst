#! /usr/bin/env python

# configure backend for matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from fitburst.analysis.peak_finder import FindPeak
from fitburst.backend.generic import DataReader
from fitburst.analysis.fitter import LSFitter
from fitburst.analysis.model import SpectrumModeler
from copy import deepcopy
import fitburst.routines.manipulate as manip
import fitburst.utilities as ut
import numpy as np
import argparse
import json
import sys
import os

parser = argparse.ArgumentParser(description=
    "A Python3 script that uses fitburst API to read, preprocess, and fit " +  
    "'generic'-formatted data against a model of the dynamic spectrum. NOTE: " + 
    "by default, the scattering timescale is not a fit parameter but can be " + 
    "turned into one through use of the --fit option."
)

parser.add_argument(
    "file", 
    action="store", 
    type=str,
    help="A Numpy state file containing data and metadata in " + 
        "fitburst-compliant ('generic') format."
)

parser.add_argument(
    "--amplitude",
    action="store",
    dest="amplitude",
    default=None,
    nargs="+",
    type=float,
    help="Initial guess for log (base 10) of the overall amplitude."
)

parser.add_argument(
    "--arrival_time",
    action="store",
    dest="arrival_time",
    default=None,
    nargs="+",
    type=float,
    help="Initial guess for arrival time, in units of seconds."
)

parser.add_argument(
    "--dm",
    action="store",
    dest="dm",
    default=None,
    nargs="+",
    type=float,
    help="Initial guess for DM, in units of pc/cc."
)

parser.add_argument(
    "--dm_index",
    action="store",
    dest="dm_index",
    default=None,
    nargs="+",
    type=float,
    help="Initial guess for index DM time delay, in units of dex."
)

parser.add_argument(
    "--downsample_freq",
    action="store",
    dest="factor_freq_downsample",
    default=1,
    type=int,
    help="Downsample the raw spectrum along the frequency axis " + 
        "by a specified integer."
)

parser.add_argument(
    "--downsample_time",
    action="store",
    dest="factor_time_downsample",
    default=1,
    type=int,
    help="Downsample the raw spectrum along the time axis by " + 
        "a specified integer."
)

parser.add_argument(
    "--fit", 
    action="store", 
    dest="parameters_to_fit", 
    default=[], 
    nargs="+", 
    type=str,
    help="A list of assumed-fixed model parameters to fit during " + 
        "least-squares estimation."
)

parser.add_argument(
    "--fix", 
    action="store", 
    dest="parameters_to_fix", 
    default=[], 
    nargs="+", 
    type=str,
    help="A list of model parameters to hold fixed to initial values."
)

parser.add_argument(
    "--folded", 
    action="store_true", 
    dest="is_folded", 
    default=False, 
    help="If set, then fit spectrum of a folded profile (e.g., " + 
        "for a pulsar observation)."
)

parser.add_argument(
    "--iterations", 
    action="store", 
    dest="num_iterations", 
    default=1, 
    type=int,
    help="Integer number of fit iterations."
)

parser.add_argument(
    "--outfile",
    action="store_true",
    dest="use_outfile_substring",
    default=False,
    help="If set, then use substring to uniquely label output " + 
        "filenamese based on input filenames."
)

parser.add_argument(
    "--peakfind_dist", 
    action="store", 
    dest="peakfind_dist", default=5, 
    type=int,
    help="Separation used for peak-finding algorithm (for multi-component fitting)."
)

parser.add_argument(
    "--peakfind_rms", 
    action="store", 
    dest="peakfind_rms", 
    default=None, 
    type=float,
    help="RMS used for peak-finding algorithm (for multi-component fitting)."
)

parser.add_argument(
    "--preprocess",
    action="store_true",
    dest="preprocess_data",
    help="If set, the run preprocessing return to normalize data and mask bad frequencies."
)

parser.add_argument(
    "--ref_freq",
    action="store",
    default=None,
    type=float,
    help="If set, then replace reference frequency with command-line value."
)

parser.add_argument(
    "--remove_smearing",
    action="store_true",
    dest="remove_dispersion_smearing",
    help="If set, then allow for removal of dispersion smearing."
)

parser.add_argument(
    "--scattering_timescale", 
    action="store", 
    dest="scattering_timescale",
    default=None, 
    nargs="+", 
    type=float,
    help="Initial guess for scattering timescale, in units of seconds."
)

parser.add_argument(
    "--scintillation",
    action="store_true",
    dest="scintillation",
    help="If set, then allow amplitude-independent modelling to account for scintillation."
)

parser.add_argument(
    "--spectral_index",
    action="store",
    dest="spectral_index",
    default=None,
    nargs="+",
    type=float,
    help="Initial guess for power-law spectral index."
)

parser.add_argument(
    "--spectral_running",
    action="store",
    dest="spectral_running",
    default=None,
    nargs="+",
    type=float,
    help="Initial guess for power-law spectral 'running'."
)

parser.add_argument(
    "--solution", 
    action="store", 
    default=None, 
    dest="solution_file", 
    type=str,
    help="If set, use existing solution in fitburst-compliant JSON format."
)

parser.add_argument(
    "--upsample_freq",
    action="store",
    dest="factor_freq_upsample",
    default=1,
    type=int,
    help="Upsample the raw spectrum along the frequency axis " + 
        "by a specified integer."
)

parser.add_argument(
    "--upsample_time",
    action="store",
    dest="factor_time_upsample",
    default=1,
    type=int,
    help="Upsample the raw spectrum along the time axis by " + 
        "a specified integer."
)

parser.add_argument(
    "--variance_range",
    action="store",
    default=[0.2, 0.8],
    dest="variance_range",
    nargs=2,
    type=float,
    help="Bounds of per-channel variance used to designate 'bad' " + 
        "channels in preprocessing step."
)

parser.add_argument(
    "--verbose", 
    action="store_true",
    default=False, 
    dest="verbose",
    help="If set, then print additional information related to " + 
        "fit parameters and fitting."
)

parser.add_argument(
    "--weight_range", 
    action="store", 
    dest="weight_range", 
    default=None, 
    nargs=2, 
    type=int,
    help="Indeces for timestamp array that represent region to evaluate RMS data weights."
)

parser.add_argument(
    "--width", 
    action="store", 
    dest="width", 
    default=None, 
    nargs="+", 
    type=float,
    help="Initial guess for burst width, in seconds."
)

parser.add_argument(
    "--window", 
    action="store", 
    dest="window", 
    default=None, 
    type=float,
    help="Half of size of data window, in seconds."
)

# grab CLI inputs from argparse.
args = parser.parse_args()
input_file = args.file
amplitude = args.amplitude
arrival_time = args.arrival_time
dm = args.dm
dm_index = args.dm_index
factor_freq_downsample = args.factor_freq_downsample
factor_time_downsample = args.factor_time_downsample
factor_freq_upsample = args.factor_freq_upsample
factor_time_upsample = args.factor_time_upsample
is_folded = args.is_folded
num_iterations = args.num_iterations
parameters_to_fit = args.parameters_to_fit
parameters_to_fix = args.parameters_to_fix
peakfind_rms = args.peakfind_rms
peakfind_dist = args.peakfind_dist
preprocess_data = args.preprocess_data
ref_freq = args.ref_freq
remove_dispersion_smearing = args.remove_dispersion_smearing
use_outfile_substring = args.use_outfile_substring
scattering_timescale = args.scattering_timescale
scintillation = args.scintillation
spectral_index = args.spectral_index
spectral_running = args.spectral_running
solution_file = args.solution_file
variance_range = args.variance_range
verbose = args.verbose
weight_range = args.weight_range
width = args.width
window = args.window

# before proceeding, adjust fixed-parameter list if necessary.
parameters_to_fix += ["dm_index", "scattering_index", "scattering_timescale"]

for current_fit_parameter in parameters_to_fit:
    if current_fit_parameter in parameters_to_fix:
        parameters_to_fix.remove(current_fit_parameter)

# also, check if JSON solution can be read (if provied.
existing_results = None

if solution_file is not None and os.path.isfile(solution_file):
    try:
        existing_results = json.load(open(solution_file, "r"))

    except:
        print("WARNING: input solution cannot be read; using basic initial parameters...")

else:
    print("INFO: no solution file found or provided; proceeding with fit...")

# also create filename substring if desired.
outfile_substring = ""

if use_outfile_substring:
    elems = input_file.split(".")
    outfile_substring = "_" + ".".join(elems[0:len(elems)-1])

# read in input data.
data = DataReader(input_file)

# load spectrum data into memory and pre-process, and load in parameter data..
data.load_data()
print(f"INFO: there are {data.num_freq} frequencies and {data.num_time} time samples.")

data.good_freq = np.sum(data.data_weights, axis=1) != 0.
data.good_freq = np.sum(data.data_full, axis=1) != 0.

# just to be sure, loop over data and ensure channels aren't "bad".
for idx_freq in range(data.num_freq):
    if data.good_freq[idx_freq]:
        if data.data_full[idx_freq, :].min() == data.data_full[idx_freq, :].max():
            print(f"ERROR: bad data value of {data.data_full[idx_freq, :].min()} in channel {idx_freq}!")
            data.good_freq[idx_freq] = False

if preprocess_data:
    data.preprocess_data(normalize_variance=True, variance_range=variance_range)

print(f"INFO: there are {data.good_freq.sum()} good frequencies...")

# now downsample after preprocessing, if desired.
data.downsample(factor_freq_downsample, factor_time_downsample)

# check if any initial guesses are missing, and overload 'basic' guess value if so.
initial_parameters = data.burst_parameters
num_components = len(initial_parameters["dm"])
basic_parameters = {
    "amplitude"        : [-2.0],
    "arrival_time"     : [np.mean(data.times)],
    "burst_width"      : [0.05],
    "dm"               : [0.0],
    "dm_index"         : [-2.0],
    "ref_freq"         : [np.min(data.freqs)],
    "scattering_index" : [-4.0], 
    "spectral_index"   : [0.0],
    "spectral_running" : [0.0],
}

for current_parameter in initial_parameters.keys():
    current_list = initial_parameters[current_parameter]

    if len(current_list) == 0:
        print(f"WARNING: parameter '{current_parameter}' has no value in data file, overloading a basic guess...")
        initial_parameters[current_parameter] = basic_parameters[current_parameter] * num_components

# now see if any parameters are missing in the dictionary.
for current_parameter in basic_parameters.keys():
    if current_parameter not in initial_parameters:
        initial_parameters[current_parameter] = basic_parameters[current_parameter] * num_components

current_parameters = deepcopy(initial_parameters)

# update DM value to use ("full" or DM offset) for dedispersion if 
# input data are already dedispersed or not.
dm_incoherent = initial_parameters["dm"][0]

if data.is_dedispersed:
    print("INFO: input data cube is already dedispersed!")
    print("INFO: setting 'dm' entry to 0, now considered a dm-offset parameter...")
    current_parameters["dm"] = [0.0] * len(initial_parameters["dm"])

if not remove_dispersion_smearing:
    dm_incoherent = 0.

# if an existing solution is supplied in a JSON file, then read it or use basic guesses.
if existing_results is not None:
    current_parameters = existing_results["model_parameters"]
    num_components = len(current_parameters["dm"])

# if values are supplied at command line, then overload those here.
if amplitude is not None:
    current_parameters["amplitude"] = amplitude

if arrival_time is not None:
    current_parameters["arrival_time"] = arrival_time

if dm is not None:
    current_parameters["dm"] = dm

if dm is not None:
    current_parameters["dm_index"] = dm_index

if scattering_timescale is not None:
    current_parameters["scattering_timescale"] = scattering_timescale

if spectral_index is not None:
    current_parameters["spectral_index"] = spectral_index

if spectral_running is not None:
    current_parameters["spectral_running"] = spectral_running

if width is not None:
    current_parameters["burst_width"] = width

# now replace ref_freq value, if desired.
if ref_freq is not None:
    current_parameters["ref_freq"] = [ref_freq] * num_components

# print parameter info if desired.
if verbose:
    print(f"INFO: initial guess for {len(current_parameters['dm'])}-component model:")

    for current_parameter_label in current_parameters.keys():
        current_list = current_parameters[current_parameter_label]
        print(f"    * {current_parameter_label}: {current_list}")

# now compute dedisperse matrix for data, given initial DM (or DM offset), 
# and grab windowed data.
print("INFO: computing dedispersion-index matrix")
data.dedisperse(
    initial_parameters["dm"][0],
    current_parameters["arrival_time"][0],
    ref_freq=initial_parameters["ref_freq"][0],
    dm_offset=0.0
)

data_windowed = data.data_full
times_windowed = data.times

if window is not None:
    data_windowed, times_windowed = data.window_data(
        current_parameters["arrival_time"][0], 
        window=window
    )

# before instantiating model, run peak-finding algorithm if desired.
if peakfind_rms is not None:
    print("INFO: running FindPeak to isolate burst components...")
    peaks = FindPeak(data_windowed, times_windowed, data.freqs, rms=peakfind_rms)
    peaks.find_peak(distance=peakfind_dist)
    current_parameters = peaks.get_parameters_dict(current_parameters)
    num_components = len(current_parameters["dm"])

# now create initial model.
print("INFO: initializing model")
model = SpectrumModeler(
    data.freqs,
    times_windowed,
    dm_incoherent = dm_incoherent,
    factor_freq_upsample = factor_freq_upsample,
    factor_time_upsample = factor_time_upsample,
    num_components = num_components,
    is_dedispersed = data.is_dedispersed,
    is_folded = is_folded,
    scintillation = scintillation,
    verbose = verbose
)
model.update_parameters(current_parameters)

# now set up fitter and execute least-squares fitting
for current_iteration in range(num_iterations):
    fitter = LSFitter(data_windowed, model, data.good_freq, weighted_fit=True, weight_range=weight_range)
    fitter.fix_parameter(parameters_to_fix)
    fitter.fit(exact_jacobian=True)

    # extract best-fit data for next loop.
    if fitter.results.success:
        bestfit_params = fitter.fit_statistics["bestfit_parameters"]
        model.update_parameters(bestfit_params)
        current_params = model.get_parameters_dict()

        if not any([x == "dm" for x in parameters_to_fix]):
            current_params["dm"] = [x for x in bestfit_params["dm"] * num_components]

        if "scattering_timescale" not in parameters_to_fix:
            current_params["scattering_timescale"] = [x for x in 
                                                      bestfit_params["scattering_timescale"] * num_components] 

        # if this is the last iteration, create best-fit model and plot windowed data.
        if current_iteration == (num_iterations - 1):
            bestfit_parameters = fitter.fit_statistics["bestfit_parameters"]
            bestfit_uncertainties = fitter.fit_statistics["bestfit_uncertainties"]
            model.update_parameters(bestfit_parameters)
            bestfit_model = model.compute_model(data=data_windowed)
            bestfit_residuals = data_windowed - bestfit_model

            if verbose:
                print(f"INFO: best-fit estimate for {len(current_parameters['dm'])}-component model:")

                for current_parameter_label in bestfit_parameters.keys():
                    current_list = bestfit_parameters[current_parameter_label]
                    current_uncertainties = bestfit_uncertainties[current_parameter_label]
                    print(f"    * {current_parameter_label}: {current_list} +/- {current_uncertainties}")        

                print("INFO: ratio of hessian matrix (approximate / exact):")
                print(fitter.hessian / fitter.hessian_approx)

            # now create plots.
            filename_elems = input_file.split(".")
            output_string = ".".join(filename_elems[:len(filename_elems)-1])
            data_grouped = ut.plotting.compute_downsampled_data(
                times_windowed, data.freqs, data_windowed, data.good_freq,
                spectrum_model = bestfit_model, factor_freq = factor_freq_downsample,
                factor_time = factor_time_downsample
            )

            ut.plotting.plot_summary_triptych(
                data_grouped, output_name = f"summary_plot{outfile_substring}.png", 
                show = False
            )

            with open(f"results_fitburst{outfile_substring}.json", "w") as out:
                json.dump(
                    {
                        "initial_dm": initial_parameters["dm"][0],
                        "initial_time": data.times_bin0,
                        "model_parameters": current_params,
                        "fit_statistics": fitter.fit_statistics,
                        "fit_logistics" : {
                            "weight_range" : weight_range,
                        }
                    },
                    out,
                    indent=4
                )
