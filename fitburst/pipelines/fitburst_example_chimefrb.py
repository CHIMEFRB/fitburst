#! /user/bin/env python

from fitburst.analysis.fitter import LSFitter
import fitburst.backend.chimefrb as chimefrb
import fitburst.analysis.model as mod
import chime_frb_constants as const
import fitburst.utilities as ut
import fitburst.routines as rt
import numpy as np
import copy
import json
import sys
import os

### import and configure matplotlib.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

### import and configure logger.
import datetime
import logging
right_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
logging.basicConfig(filename=f"fitburst_run_{right_now}.log", level=logging.DEBUG)
log = logging.getLogger("fitburst")

### import and configure argparse.
import argparse

parser = argparse.ArgumentParser(description=
    "A Python3 script that uses fitburst API to read, preprocess, window, and fit CHIME/FRB data " + 
    "against a model of the dynamic spectrum." 
)

parser.add_argument(
    "eventIDs", action="store", nargs="+", type=int,
    help="One or more CHIME/FRB event IDs."
)

parser.add_argument(
    "--amplitude", action="store", dest="amplitude", default=None, nargs="+", type=float,
    help="Initial guess for burst amplitude, in dex."
)

parser.add_argument(
    "--dm", action="store", dest="dm", default=None, nargs="+", type=float,
    help="Initial guess for dispersion measure (DM), in pc/cc."
)

parser.add_argument(
    "--fit", action="store", dest="parameters_to_fit", default=[], nargs="+", type=str,
    help="A list of model parameters to fit during least-squares estimation."
)

parser.add_argument(
    "--fix", action="store", dest="parameters_to_fix", default=[], nargs="+", type=str,
    help="A list of model parameters to hold fixed to initial values."
)

parser.add_argument(
    "--iterations", action="store", dest="num_iterations", default=1, type=int,
    help="Integer number of fit iterations."
)

parser.add_argument(
    "--latest", action="store", default=None, dest="latest_solution_location", type=str,
    help="If set, use existing solution if present."
)

parser.add_argument(
    "--offset_dm", action="store", dest="offset_dm", default=0.0, type=float,
    help="Offset applied to initial dispersion measure, in pc/cc."
)

parser.add_argument(
    "--offset_time", action="store", dest="offset_time", default=0.0, type=float,
    help="Offset applied to initial arrival time, in seconds."
)

parser.add_argument(
    "--pipeline", action="store", dest="pipeline", default="L1", type=str,
    help="Name of CHIME/FRB pipeline whose results will be used as initial guesses."
)

parser.add_argument(
    "--save", action="store_true", dest="save_results",
    help="If set, save best-fit results to a JSON file."
)

parser.add_argument(
    "--scattering_timescale", action="store", dest="scattering_timescale", default=None, nargs="+", type=float,
    help="Initial guess for scattering index."
)

parser.add_argument(
    "--spectral_index", action="store", dest="spectral_index", default=None, nargs="+", type=float,
    help="Initial guess for spectral index."
)

parser.add_argument(
    "--spectral_running", action="store", dest="spectral_running", default=None, nargs="+", type=float,
    help="Initial guess for spectral running."
)

parser.add_argument(
    "--width", action="store", dest="width", default=None, nargs="+", type=float,
    help="Initial guess for burst width, in seconds."
)

parser.add_argument(
    "--window", action="store", dest="window", default=0.08, type=float,
    help="Half of size of data window, in seconds."
)

### grab CLI inputs from argparse.
args = parser.parse_args()
eventIDs = args.eventIDs
amplitude = args.amplitude
dm = args.dm
latest_solution_location = args.latest_solution_location
num_iterations = args.num_iterations
offset_dm = args.offset_dm
offset_time = args.offset_time
parameters_to_fit = args.parameters_to_fit
parameters_to_fix = args.parameters_to_fix
pipeline = args.pipeline
save_results = args.save_results
scattering_timescale = args.scattering_timescale
spectral_index = args.spectral_index
spectral_running = args.spectral_running
width = args.width
window_orig = args.window


### before looping over events, suss out model parameters to fit and/or hold fixed.
parameters_to_fix += ["dm_index", "scattering_index", "scattering_timescale"]

for current_fit_parameter in parameters_to_fit:
    if current_fit_parameter in parameters_to_fix:
        parameters_to_fix.remove(current_fit_parameter)

### loop over all CHIME/FRB events supplied at command line.
for current_event_id in eventIDs:

    # grab initial parameters to check if pipeline-specific parameters exist.
    data = chimefrb.DataReader(current_event_id)
    print(data.burst_parameters)
    initial_parameters = data.get_parameters(pipeline=pipeline)
        
    # if returned parameter dictionary is empty, move on to next event.
    if bool(initial_parameters):
        log.info(f"successfully grabbed parameter data from CHIME/FRB {pipeline} pipeline")
        
    else:
        log.error(f"couldn't grab CHIME/FRB {pipeline} pipeline data for event {current_event_id}")
        continue

    window = window_orig

    # load data into memory and pre-process.
    try:
        data.load_data(data.files)
        log.info(f"successfully read raw msgpack data for event {current_event_id}")

    except Exception as exc:
        log.error(f"couldn't read raw msgpack data for event {current_event_id}")
        continue

    # now that frame0-nano value is available after loading of data, grab parameters 
    # to obtain timestamp info.
    initial_parameters = data.get_parameters(pipeline=pipeline)
    # if a JSON file containing results already exists, then read that in.
    latest_solution_file = f"{latest_solution_location}/results_fitburst_{current_event_id}.json"

    if (
        latest_solution_location is not None and os.path.isfile(latest_solution_file)
    ):
        log.info(f"loading data from results file for event {current_event_id}")
        results = json.load(open(latest_solution_file, "r"))
        initial_parameters = results["model_parameters"]
        log.info("window size adjusted to +/- {0:.1f} ms".format(window * 1e3))

    elif (
        latest_solution_location is not None and os.path.isfile(latest_solution_file)
    ):
        log.info(f"results already obtained and saved for {current_event_id}; ignoring fit and moving on...")
        continue

    else: 
        pass
        #initial_parameters["burst_width"] = [window / 10.]

    # if scattering timescale is a fit parameter, initially set to width.
    if (
        initial_parameters["scattering_timescale"][0] == 0. and 
        "scattering_timescale" not in parameters_to_fix
    ):
        initial_parameters["scattering_timescale"] = copy.deepcopy(
            (np.array(initial_parameters["burst_width"]) * 3.).tolist()
        )
        initial_parameters["burst_width"] = (np.array(initial_parameters["burst_width"]) / 3.).tolist()

    # if guesses are provided at command, overload them into the initial-guess dictionary.
    initial_parameters["dm"][0] += offset_dm
    initial_parameters["arrival_time"][0] += offset_time

    if amplitude is not None:
        initial_parameters["amplitude"] = amplitude
   
    if dm is not None:
        initial_parameters["dm"] = dm
   
    if scattering_timescale is not None:
        initial_parameters["scattering_timescale"] = scattering_timescale

    if spectral_index is not None:
        initial_parameters["spectral_index"] = spectral_index

    if spectral_running is not None:
        initial_parameters["spectral_running"] = spectral_running

    if width is not None:
        initial_parameters["burst_width"] = width
   
    # now, clean and normalize data.
    data.preprocess_data(
        normalize_variance = False,
        variance_range = [0.95, 1.05],
        variance_weight=(1. / const.L0_NUM_FRAMES_SAMPLE / 2)
    )

    # if the number of RFI-flagged channels is "too large", skip this event altogether.
    num_bad_freq = data.num_freq - np.sum(data.good_freq)

    if (num_bad_freq / data.num_freq) > 0.7:
        log.error(
            f" {num_bad_freq} out of {data.num_freq} frequencies masked for event {current_event_id}"
        )
        continue

    # now compute dedisperse matrix for data, given initial DM, and grab windowed data.
    print("INFO: computing dedispersion-index matrix")
    print("INFO: dedispersing over freq range ({0:.3f}, {1:.3f}) MHz".format(
            np.min(data.freqs), np.max(data.freqs)
        )
    )
    params = initial_parameters#data.burst_parameters["fitburst"]["round_2"]
    data.dedisperse(
        params["dm"][0],
        params["arrival_time"][0],
        reference_freq=params["ref_freq"][0]
    )

    # before doing anything, check if window size doesn't extend beyond data set.
    # if it does, adjust down by an appropriate amount.
    window_max = data.times[-1] - initial_parameters["arrival_time"][0]

    if window > window_max:
        window = window_max - 0.001
        print("INFO: window size adjusted to +/- {0:.1f} ms".format(window * 1e3))

    data_windowed, times_windowed = data.window_data(params["arrival_time"][0], window=window)

    # now create initial model.
    # since CHIME/FRB data are in msgpack format, define a few things 
    # so that this version of fitburst works similar to the original version on site.
    print("INFO: initializing model")
    model = mod.SpectrumModeler()
    model.is_dedispersed = False
    model.set_dimensions(data.num_freq, len(times_windowed))
    model.set_dedispersion_idx(data.dedispersion_idx)
    model.update_parameters(initial_parameters)

    ### now set up fitter and execute least-squares fitting.
    for current_iteration in range(num_iterations):
        print(f"INFO: fitting model, loop #{current_iteration + 1}")
        fitter = LSFitter(model)
        fitter.fix_parameter(parameters_to_fix)
        fitter.weighted_fit = True
        fitter.fit(data.times, data.freqs, data_windowed)
    
        # before executing the fitting loop, overload model class with best-fit parameters.
        if fitter.success:
            model.update_parameters(fitter.fit_statistics["bestfit_parameters"])

    ### now compute best-fit model of spectrum and plot.
    if fitter.success:
        bestfit_model = model.compute_model(data.times, data.freqs) * data.good_freq[:, None]
        bestfit_residuals = data_windowed - bestfit_model

        # create summary plot.
        ut.plotting.plot_summary_triptych(
            data.times, data.freqs, data_windowed, data.good_freq, model = bestfit_model, 
            residuals = bestfit_residuals, output_name = f"summary.{current_event_id}.png",
            factor_freq = 64
        )

        # create JSON file contain burst parameters and statistics.
        with open(f"results_fitburst_{current_event_id}.json", "w") as out:
            json.dump(
                {
                    "model_parameters": model.get_parameters_dict(), 
                    "fit_statistics": fitter.fit_statistics,
                },
                out, 
                indent=4
            )

        # finally, if desired, save spectrum and burst-parameter/metadata dictionaries.
        if save_results:
            np.savez(
                f"test_data_CHIMEFRB_{current_event_id}.npz",
                burst_parameters = model.get_parameters_dict(),
                data_full = data_windowed,
                metadata = {
                    "bad_chans" : [],
                    "freqs_bin0" : data.freqs[0],
                    "is_dedispersed" : True,
                    "num_freq" : data.num_freq,
                    "num_time" : len(times_windowed),
                    "times_bin0" : times_windowed[0],
                    "res_freq" : data.res_freq,
                    "res_time" : data.res_time,
                }
                
            )
