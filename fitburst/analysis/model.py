from fitburst.backend import general
import fitburst.routines as rt
import numpy as np
import sys

class SpectrumModeler(object):
    """
    A python structure that contains all information regarding parameters that 
    describe and are used to compute models of dynamic spectra.
    """

    def __init__(self, spectrum_model="powerlaw"):

        # first define model-configuration parameters that are not fittable.
        self.dedispersion_idx = None
        self.num_components = 1
        self.num_freq = None
        self.num_time = None
        self.spectrum_model = spectrum_model

        # define the basic model parameters first.
        self.parameters_all = [
            "dm",
            "dm_index",
            "scattering_timescale",
            "scattering_index",
            "amplitude",
            "arrival_time",
            "burst_width"
        ]

        # add the appropriate spectral parameters to the list.
        if (self.spectrum_model == "powerlaw"):
            self.parameters_all += ["spectral_index", "spectral_running"]

        elif (self.spectrum_model == "gaussian"):
            self.parameters_all += ["freq_mean", "freq_width"]

        for current_parameter in self.parameters_all:
            setattr(self, current_parameter, [None])

    def compute_model(self, times: np.float, freqs: np.float):
        """
        Computes the model dynamic spectrum based on model parameters (set as class 
        attributes) and input values of times and frequencies.

        Parameters
        ----------
        times : np.ndarray
        """

        # initialize model matrix and size of temporal window.
        model_spectrum = np.zeros((self.num_freq, self.num_time), dtype=np.float)
        num_window_bins = self.num_time // 2
        # loop over all components.
        for current_component in range(self.num_components):

            # extract parameter values for current component.
            current_amplitude = self.amplitude[current_component]
            current_arrival_time = self.arrival_time[current_component]
            current_dm = self.dm[0]
            current_dm_index = self.dm_index[0]
            current_ref_freq = self.reference_freq[current_component]
            current_sc_idx = self.scattering_index[0]
            current_sc_time = self.scattering_timescale[0]
            current_width = self.burst_width[current_component]

            #print("Current parameters: ", current_dm, current_amplitude, current_arrival_time, 
                #current_sc_idx, current_sc_time, current_width, end=", ")

            # now loop over bandpass.
            for current_freq in range(self.num_freq):

                # compute dispersion-corrected timeseries.
                # first, check if model is for dispersed data.
                if not self.is_dedispersed and self.dedispersion_idx is not None:
                    current_arrival_idx = self.dedispersion_idx[current_freq]
                    current_delay = current_arrival_time + rt.ism.compute_time_dm_delay(
                        current_dm,
                        general["constants"]["dispersion"],
                        current_dm_index,
                        freqs[current_freq],
                        freq2=current_ref_freq,
                    )

                    # NOTE: the .copy() below is important!
                    current_times = times[
                        current_arrival_idx - num_window_bins: current_arrival_idx + num_window_bins
                    ].copy()
                    current_times -= current_delay

                # if data is already dedipsersed and nominal DM is specified,
                # compute "relative" DM delay.
                elif self.is_dedispersed: 
                    relative_delay = rt.ism.compute_time_dm_delay(
                        current_dm,
                        general["constants"]["dispersion"],
                        current_dm_index,
                        freqs[current_freq],
                        freq2=current_ref_freq,
                    )
                    
                    # now compute current-times array corrected for relative delay.
                    current_times = times.copy() - current_arrival_time
                    current_times -= relative_delay

                else:
                    sys.exit("ERROR: type of dedispersion plan is ambiguous!")                    

                # first, scale scattering timescale.
                current_sc_time_scaled = rt.ism.compute_time_scattering(
                    freqs[current_freq],
                    current_ref_freq,
                    current_sc_time,
                    current_sc_idx
                )

                # second, compute raw profile form.
                current_profile = self.compute_profile(
                    current_times,
                    0.0, # since 'current_times' is already corrected for DM.
                    current_sc_time_scaled,
                    current_width,
                )

                # third, compute and scale profile by spectral energy distribution.
                if (self.spectrum_model == "powerlaw"):
                    current_sp_idx = self.spectral_index[current_component]
                    current_sp_run = self.spectral_running[current_component]

                    current_profile *= rt.spectrum.compute_spectrum_rpl(
                        freqs[current_freq],
                        current_ref_freq,
                        current_sp_idx,
                        current_sp_run,
                    )

                elif (self.spectrum_model == "gaussian"):
                    pass

                # finally, add to approrpiate slice of model-spectrum matrix.
                model_spectrum[current_freq, :] += (10**current_amplitude) * current_profile

            # print spectral index/running for current component.
            #print(current_sp_idx, current_sp_run)

        return model_spectrum

    def compute_profile(self, times: np.float, arrival_time: np.float, sc_time: np.float,
        width: np.float):
        """
        Returns the Gaussian or pulse-broadened temporal profile, depending on input 
        values of width and scattering timescale..
        """

        profile = np.zeros(len(times), dtype=np.float)

        # compute either Gaussian or pulse-broadening function, depending on inputs.
        if sc_time < np.fabs(0.15 * width):
            profile = rt.profile.compute_profile_gaussian(times, arrival_time, width)

        else:
            profile = rt.profile.compute_profile_pbf(times, arrival_time, width, sc_time)

        return profile

    def compute_spectrum(self, freqs: np.float, freq_mean: np.float, freq_width: np.float,
        sp_idx: np.float, sp_run: np.float):
        """
        Returns the Gaussian or power-law spectral energy distribution, depending on the 
        desired spectral model (set by self.spectrum_model). 
        """

        spectrum = 0.

        if (self.spectrum_model == "powerlaw"):
            spectrum = rt.spectrum.compute_spectrum_rpl(
                freqs,
                current_ref_freq,
                sp_idx,
                sp_run
            )

        elif (self.spectrum_model == "gaussian"):
            spectrum = rt.spectrum.compute_spectrum_gaussian(freqs, freq_mean, freq_width)

        return spectrum

    def set_dimensions(self, num_freq, num_time):
        """
        Sets the dimensions of the model spectrum. If data are windowed in frequency 
        and/or time during preprocessing, then the inputs must match the dimensions of 
        the processed data.
        """

        self.num_freq = num_freq
        self.num_time = num_time

        print("INFO: dimensions of model spectrum set to ({0}, {1})".format(
            self.num_freq, self.num_time)
        )

    def set_dedispersion_idx(self, dedispersion_idx):
        """
        Creates or overloads an array of time bins where the dispersed pulse is observed.
        """

        self.dedispersion_idx = dedispersion_idx

        print("INFO: array of dedispersion indices loaded successfully")

    def update_parameters(self, model_parameters: dict):
        """
        Overloads parameter values stored in object with those supplied by the user.

        Parameters
        ----------
        model_parameters : dict
            a Python dictionary with parameter names listed as keys, parameter values 
            supplied as lists tied to keys.

        Returns
        -------
        None : None
            this method overloads class attributes.
        """

        # first, overload attributes with values for supplied parameters.
        for current_parameter in model_parameters.keys():
            setattr(self, current_parameter, model_parameters[current_parameter])

            # if number of components is 2 or greater, update num_components attribute.
            if len(model_parameters[current_parameter]) > 1:
                setattr(self, "num_components", len(model_parameters[current_parameter]))
