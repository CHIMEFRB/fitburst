#! /user/bin/env python

from fitburst.analysis.fitter import LSFitter
import fitburst.backend.chimefrb as chimefrb
import fitburst.analysis.model as mod
import chime_frb_constants as const
import fitburst.utilities as ut
import fitburst.routines as rt
import numpy as np
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
    "--iterations", action="store", dest="num_iterations", default=1, type=int,
    help="Integer number of fit iterations."
)

parser.add_argument(
    "--latest", action="store_true", dest="use_latest_solution",
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
    "--window", action="store", dest="window", default=0.08, type=float,
    help="Half of size of data window, in seconds."
)

parser.add_argument(
    "--width", action="store", dest="width", default=None, nargs="+", type=float,
    help="Initial guess for burst width, in seconds."
)

### grab CLI inputs from argparse.
args = parser.parse_args()
eventIDs = args.eventIDs
window_orig = args.window
num_iterations = args.num_iterations
offset_dm = args.offset_dm
offset_time = args.offset_time
pipeline = args.pipeline
use_latest_solution = args.use_latest_solution
width = args.width

### loop over all CHIME/FRB events supplied at command line.
for current_event_id in eventIDs:

    # grab initial parameters to check if pipeline-specific parameters exist.
    data = chimefrb.DataReader(current_event_id)
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
    initial_parameters["dm"][0] += offset_dm
    initial_parameters["arrival_time"][0] += offset_time

    # if a JSON file containing results already exists, then read that in.
    if use_latest_solution and os.path.isfile(f"results_fitburst_{current_event_id}.json"):
        print(f"INFO: loading data from results file for event {current_event_id}")
        results = json.load(open(f"results_fitburst_{current_event_id}.json", "r"))
        initial_parameters = results["model_parameters"]
        print("INFO: window size adjusted to +/- {0:.1f} ms".format(window * 1e3))

    else: 
        pass
        #initial_parameters["burst_width"] = [window / 10.]

    # if guesses are provided at command, overload them into the initial-guess dictionary.
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
        fitter.fix_parameter(
            [
                #"dm", 
                "dm_index", 
                "scattering_index", 
                "scattering_timescale"
            ]
        )
        fitter.weighted_fit = True
        fitter.fit(data.times, data.freqs, data_windowed)
    
        # before executing the fitting loop, overload model class with best-fit parameters.
        if fitter.success:
            model.update_parameters(fitter.fit_statistics["bestfit_parameters"])

    ### now compute best-fit model of spectrum and plot.
    if fitter.success:
        bestfit_model = model.compute_model(data.times, data.freqs) * data.good_freq[:, None]
        bestfit_residuals = data_windowed - bestfit_model

        ut.plotting.plot_summary_triptych(
            data.times, data.freqs, data_windowed, data.good_freq, model = bestfit_model, 
            residuals = bestfit_residuals, output_name = f"summary.{current_event_id}.png",
            factor_freq = 64
        )

        # finally, stash results into a JSON file.

        with open(f"results_fitburst_{current_event_id}.json", "w") as out:
            json.dump(
                {
                    "model_parameters": model.get_parameters_dict(), 
                    "fit_statistics": fitter.fit_statistics,
                },
                out, 
                indent=4
            )
