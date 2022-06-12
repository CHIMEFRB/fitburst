"""
Object for Computing and Updating Models of Dynamic Spectra

The SpectrumModeler() object is designed to compute models of dynamic
spectra based on parameter values, and handle the updating of one or more
model parameters. The updating/retrieval methods are used in the fitburst
fitter object, and are written to handle user-specified fixing of parameters.
"""
import sys
import numpy as np

from fitburst.backend import general
import fitburst.routines as rt

class SpectrumModeler:
    """
    A Python structure that contains all information regarding parameters that
    describe and are used to compute models of dynamic spectra.
    """

    # pylint: disable=too-many-instance-attributes

    def __init__(self, num_freq: int, num_time: int, dm_incoherent: float = 0.,
        freq_model: str = "powerlaw", factor_freq_upsample: int = 1,
        factor_time_upsample: int = 1, is_dedispersed: bool = False,
        is_folded: bool = False, verbose: bool = False) -> None:
        """
        Instantiates the model object and sets relevant parameters, depending on
        desired model for spectral energy distribution.

        Parameters
        ----------

        num_freq: float
            The total number of frequency channels in the spectrum (including masked ones)

        num_time: float
            The total number of time samples in the spectrum for which a model is desired

        dm_incoherent : float, optional
            The DM used to incoherently dedisperse input data; this is only used if the
            'is_dedispersed' argument is set to True

        factor_freq_upsample : int, optional
            The factor to upsample each frequency label into an array of subbands

        factor_time_upsample : int, optional
            The factor to upsample the array of timestamps

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
            If true, then print parameter values during each function call
            (This is mainly useful to gauge least-squares fitting algorithms.)

        """

        # pylint: disable=too-many-arguments,too-many-locals

        # first define model-configuration parameters that are not fittable.
        self.dedispersion_idx = None
        self.dm_incoherent = dm_incoherent
        self.factor_freq_upsample = factor_freq_upsample
        self.factor_time_upsample = factor_time_upsample
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

        # pylint: disable=no-member,too-many-locals

        # determine resolutions.
        res_freq = np.diff(freqs)[0]
        res_time = np.diff(times)[0]

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
                print(
                    f"{current_dm:.5f}  {current_amplitude:.5f}  {current_arrival_time:.5f}  ",
                    f"{current_sc_idx:.5f}  {current_sc_time:.5f}  {current_width:.5f}", end=" ")

            # now loop over bandpass.
            for current_freq in range(self.num_freq):

                # create an upsampled version of the current frequency label.
                # even if no upsampling is desired, this will return an array
                # of length 1.
                current_freq_arr = rt.manipulate.upsample_1d(
                    [freqs[current_freq]],
                    res_freq,
                    self.factor_freq_upsample
                )

                # compute dispersion-corrected timeseries.
                # first, check if model is for dispersed data.
                if not self.is_dedispersed and self.dedispersion_idx is not None:
                    current_arrival_idx = self.dedispersion_idx[current_freq]
                    current_delay = current_arrival_time + rt.ism.compute_time_dm_delay(
                        current_dm,
                        general["constants"]["dispersion"],
                        current_dm_index,
                        current_freq_arr,
                        freq2=current_ref_freq,
                    )

                    # NOTE: the .copy() below is important!
                    current_times = times[
                        current_arrival_idx - num_window_bins: current_arrival_idx + num_window_bins
                    ].copy()
                    current_times_arr, _ = np.meshgrid(current_times, current_freq_arr)
                    current_times_arr -= current_delay[:, None]

                # if data is already dedipsersed, then compute "relative" DM delay.
                elif self.is_dedispersed:

                    # first, compute "full" delays for all upsampled frequency labels.
                    relative_delay = rt.ism.compute_time_dm_delay(
                        self.dm_incoherent + current_dm,
                        general["constants"]["dispersion"],
                        current_dm_index,
                        current_freq_arr,
                        freq2=current_ref_freq,
                    )

                    # then compute "relative" delays with respect to central frequency.
                    relative_delay -= rt.ism.compute_time_dm_delay(
                        self.dm_incoherent,
                        general["constants"]["dispersion"],
                        current_dm_index,
                        freqs[current_freq],
                        freq2=current_ref_freq,
                    )

                    # now compute current-times array corrected for relative delay.
                    current_times = rt.manipulate.upsample_1d(
                        times.copy() - current_arrival_time,
                        res_time,
                        self.factor_time_upsample
                    )

                    current_times_arr, _ = np.meshgrid(current_times, current_freq_arr)
                    current_times_arr -= relative_delay[:, None]

                else:
                    sys.exit("ERROR: type of dedispersion plan is ambiguous!")

                # first, adjust scattering timescale to current frequency label(s).
                current_sc_time_scaled = rt.ism.compute_time_scattering(
                    current_freq_arr,
                    current_ref_freq,
                    current_sc_time,
                    current_sc_idx
                )

                # second, compute raw temporal profile.
                current_profile = self.compute_profile(
                    current_times_arr,
                    0.0, # since 'current_times' is already corrected for DM.
                    current_sc_time_scaled,
                    current_width,
                    is_folded = self.is_folded,
                )

                # third, compute and scale profile by spectral energy distribution.
                if self.freq_model == "powerlaw":
                    current_sp_idx = self.spectral_index[current_component]
                    current_sp_run = self.spectral_running[current_component]

                    current_profile *= rt.spectrum.compute_spectrum_rpl(
                        current_freq_arr,
                        current_ref_freq,
                        current_sp_idx,
                        current_sp_run,
                    )[:, None]

                elif self.freq_model == "gaussian":
                    current_freq_mean = self.freq_mean[current_component]
                    current_freq_width = self.freq_width[current_component]

                    current_profile *= rt.profile.compute_profile_gaussian(
                        current_freq_arr,
                        current_freq_mean,
                        current_freq_width,
                    )[:, None]

                # before writing, downsize upsampled array to original size.
                current_profile = rt.manipulate.downsample_1d(
                    current_profile.mean(axis=0),
                    self.factor_time_upsample
                )

                # finally, add to approrpiate slice of model-spectrum matrix.
                model_spectrum[current_freq, :] += (10**current_amplitude) * current_profile

            # print spectral index/running for current component.
            if self.verbose:
                if self.freq_model == "powerlaw":
                    print(f"{current_sp_idx:.5f}  {current_sp_run:.5f}")

                elif self.freq_model == "gaussian":
                    print(f"{current_freq_mean:.5f}  {current_freq_width:.5f}")


        return model_spectrum

    def compute_profile(self, times: float, arrival_time: float, sc_time: float,
        width: float, is_folded: bool = False) -> float:
        """
        Returns the temporal profile, depending on input values of width
        and scattering timescale.

        Parameters
        ----------
        times : float
            One or more values corresponding to time

        arrival_time : float
            The arrival time of the burst

        sc_time : float
            The scattering timescale of the burst (which depends on frequency label)

        width : float
            The intrinsic temporal width of the burst

        is_folded : bool, optional
            If true, then the temporal profile is computed over two realizations and then
            averaged down to one (in order to allow for wrapping of a folded pulse shape)

        Returns
        -------
        profile : float
            One or more values of the temporal profile, evaluated at the input timestamps
        """

        # pylint: disable=too-many-arguments,no-self-use

        # if data are "folded" (i.e., data from pulsar timing observations),
        # model at twice the timespan and wrap/average the two realizations.
        # this step is to account for potential wrapping of pulse shape.
        times_copy = times.copy()

        if is_folded:
            res_time = np.unique(np.diff(times_copy))
            times_copy = np.append(times,
                np.linspace(1, len(times), num=len(times)) * res_time + times[-1])

        # compute either Gaussian or pulse-broadening function, depending on inputs.
        profile = np.zeros(len(times_copy), dtype=np.float)

        if np.any(sc_time < np.fabs(0.15 * width)):
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

        Parameters
        ----------
        None : NoneType
            this method uses existing class attributes

        Returns
        -------
        parameter_dict : dict
            A dictionary containing parameter names as keys, and lists of per-component
            values as the dictionary values.
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
        This method is meant to separate an eventual 'legacy' method of dedispersion that
        is current used by CHIME/FRB.

        Parameters
        ----------
        dedispersion_idx : int
            An array of integers that represent indeces where a dipsersed signal is
            expected in filterbank data

        Returns
        -------
        None : None
            this method overloads class attributes.
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
