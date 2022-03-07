from fitburst.backend import general
import fitburst.routines as rt
import numpy as np
import sys

class SpectrumModeler(object):
    """
    A Python structure that contains all information regarding parameters that 
    describe and are used to compute models of dynamic spectra.
    """

    def __init__(self, num_freq: int, num_time: int, freq_model: str = "powerlaw", 
        is_dedispersed: bool = False, is_folded: bool = False, verbose: bool = False):
        """
        Instantiates the model object and sets relevant parameters, depending on 
        desired model for spectral energy distribution.

        Parameters
        ----------

        num_freq: float
            The total number of frequency channels in the spectrum (including masked ones)

        num_time: float
            The total number of time samples in the spectrum for which a model is desired

        freq_model : str, optional
            The name of the desired spectral energy distribution, currently either 
            'gaussian' or 'powerlaw'

        is_dedispersed : bool, optional
            If true, then assume that the dispersion measure is an 'offset' parameter 
            and computes the relative dispersion for non-zero offset values

        is_folded : bool, optional
            If true, then the temporal profile is computed over two realizations and then 
            averaged down to one (in order to allow for wrapping of a folded pulse shape)

        verbose : bool, optional
            If true, then print parameter values during each function call. 
            (This is mainly useful to gauge least-squares fitting algorithms.)

        """

        # first define model-configuration parameters that are not fittable.
        self.dedispersion_idx = None
        self.freq_model = freq_model
        self.is_dedispersed = is_dedispersed
        self.is_folded = is_folded
        self.num_components = 1
        self.num_freq = num_freq
        self.num_time = num_time
        self.verbose = verbose

        # define all *fittable* model parameters first.
        # NOTE: 'ref_freq' is not listed here as it's a parameter that is always held fixed.
        self.parameters = [
            "amplitude",
            "arrival_time",
            "burst_width",
            "dm",
            "dm_index",
            "scattering_timescale",
            "scattering_index",
        ]

        # add the appropriate spectral parameters to the list.
        if self.freq_model == "powerlaw":
            self.parameters += ["spectral_index", "spectral_running"]

        elif self.freq_model == "gaussian":
            self.parameters += ["freq_mean", "freq_width"]

        else:
            sys.exit(f"ERROR: cannot recognize SED model of type '{freq_model}")

        # now instantiate parameter attributes and set initially to NoneType.
        for current_parameter in self.parameters:
            setattr(self, current_parameter, None)

    def compute_model(self, times: float, freqs: float) -> float:
        """
        Computes the model dynamic spectrum based on model parameters (set as class 
        attributes) and input values of times and frequencies.

        Parameters
        ----------
        times : np.ndarray
            an array of values corresponding to time

        freqs : np.ndarray
            an array of values corresponding to observing frequency
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
            current_ref_freq = self.ref_freq[current_component]
            current_sc_idx = self.scattering_index[0]
            current_sc_time = self.scattering_timescale[0]
            current_width = self.burst_width[current_component]

            if self.verbose:
                print("{0:.5f}  {1:.5f}  {2:.5f}  {3:.5f}  {4:.5f} {5:.5f}".format(
                    current_dm, current_amplitude, current_arrival_time, 
                    current_sc_idx, current_sc_time, current_width), end=" "
                )

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
                    is_folded = self.is_folded,
                )

                # third, compute and scale profile by spectral energy distribution.
                if (self.freq_model == "powerlaw"):
                    current_sp_idx = self.spectral_index[current_component]
                    current_sp_run = self.spectral_running[current_component]

                    current_profile *= rt.spectrum.compute_spectrum_rpl(
                        freqs[current_freq],
                        current_ref_freq,
                        current_sp_idx,
                        current_sp_run,
                    )

                elif (self.freq_model == "gaussian"):
                    current_freq_mean = self.freq_mean[current_component]
                    current_freq_width = self.freq_width[current_component]

                    current_profile *= rt.profile.compute_profile_gaussian(
                        freqs[current_freq],
                        current_freq_mean,
                        current_freq_width,
                    )

                # finally, add to approrpiate slice of model-spectrum matrix.
                model_spectrum[current_freq, :] += (10**current_amplitude) * current_profile

            # print spectral index/running for current component.
            if self.verbose:
                if self.freq_model == "powerlaw":
                    print("{0:.5f}  {1:.5f}".format(current_sp_idx, current_sp_run))

                elif self.freq_model == "gaussian":
                    print("{0:.5f}  {1:.5f}".format(current_freq_mean, current_freq_width))


        return model_spectrum

    def compute_profile(self, times: float, arrival_time: float, sc_time: float,
        width: float, is_folded: bool = False) -> float:
        """
        Returns the Gaussian or pulse-broadened temporal profile, depending on input 
        values of width and scattering timescale..
        """

        # if data are "folded" (i.e., data from pulsar timing observations),
        # model at twice the timespan and wrap/average the two realizations. 
        # this step is to account for potential wrapping of pulse shape.
        times_copy = times.copy()
        res_time = times_copy[1] - times_copy[0]

        if is_folded:
            times_copy = np.append(
                times, 
                np.linspace(1, len(times), num=len(times)) * res_time + times[-1]
            )

        # compute either Gaussian or pulse-broadening function, depending on inputs.
        profile = np.zeros(len(times_copy), dtype=np.float)

        if sc_time < np.fabs(0.15 * width):
            profile = rt.profile.compute_profile_gaussian(times_copy, arrival_time, width)

        else:
            # the following times array manipulates the times array so that we avoid a 
            # floating-point overlow in the exp((-times - toa) / sc_time) term in the 
            # PBF call. TODO: use a better, more transparent method for avoiding this.
            times_copy[times_copy < -5 * width] = -5 * width
 
            # now call the function.
            profile = rt.profile.compute_profile_pbf(times_copy, arrival_time, width, sc_time)

        if is_folded:
            profile = np.sum(profile.reshape(2, len(times)), axis=0) / 2

        return profile

    def get_parameters_dict(self) -> dict:
        """
        Returns model parameters as a dictionary, with keys set to the parameter names 
        and values set to the Python list containing parameter values.
        """

        parameter_dict = {}

        # loop over all fittable parameters and grab their values.
        for current_parameter in self.parameters:
            parameter_dict[current_parameter] = getattr(self, current_parameter)

        # before exiting, grab the values of the reference frequency, which 
        # isn't fittable and is therefore not in the 'parameters' list.
        parameter_dict["ref_freq"] = getattr(self, "ref_freq")

        return parameter_dict

    def set_dedispersion_idx(self, dedispersion_idx: int) -> None:
        """
        Creates or overloads an array of time bins where the dispersed pulse is observed.
        """

        self.dedispersion_idx = dedispersion_idx

        print("INFO: array of dedispersion indices loaded successfully")

    def update_parameters(self, model_parameters: dict) -> None:
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
