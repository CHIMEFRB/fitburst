import fitburst.routines as rt
import numpy as np

class SpectrumModeler(object):
    """
    A python structure that contains all information regarding parameters that 
    describe and are used to compute models of dynamic spectra.
    """

    def __init__(self):
        # define the basic model parameters first.
        self.parameters_all = [
            "amplitude",
            "arrival_time",
            "dm",
            "freq_mean",
            "freq_width",
            "scattering_index",
            "scattering_timescale",
            "spectral_index",
            "spectral_running",
            "width"
        ]

        for current_parameter in self.parameters_all:
            setattr(self, current_parameter, [None])

        #self.amplitude = [None]
        #self.arrival_time = [None]
        #self.dm = [None]
        #self.freq_mean = [None]
        #self.freq_width = [None]
        #self.scattering_index = [None]
        #self.scattering_timescale = [None]
        #self.spectral_index = [None]
        #self.spectral_running = [None]
        #self.width = [None]

        # now define model-configuration parameters that are not fittable.
        self.num_components = 1
        self.reference_freq = None
        self.spectrum_model = "powerlaw"

    def compute_model(self, times: np.float, freqs: np.float):
        """
        Computes the model dynamic spectrum based on model parameters (set as class 
        attributes) and input values of times and frequencies.

        Parameters
        ----------
        times : np.ndarray
        """

        model_spectrum = np.zeros((len(freqs), len(times)), dtype=np.float)

        # loop over all components.
        for current_component in range(self.num_components):
            # extract parameter values for current component.
            current_amplitude = self.amplitude[current_component]
            current_arrival_time = self.arrival_time[current_component]
            current_freq_mean = self.freq_mean[current_component]
            current_freq_width = self.freq_width[current_component]
            current_sc_idx = self.scattering_index[current_component]
            current_sc_time = self.scattering_timescale[current_component]
            current_sp_idx = self.spectral_index[current_component]
            current_sp_run = self.spectral_running[current_component]
            current_width = self.width[current_component]

            # now loop over bandpass.
            for current_freq in range(len(freqs)):
                # first, scale scattering timescale.
                current_sc_time_scaled = rt.ism.compute_time_scattering(
                    self.freqs[current_freq],
                    self.reference_freq,
                    current_sc_time,
                    sc_idx=current_sc_idx
                )

                # second, compute raw profile form.
                current_profile = self.compute_profile(
                    times,
                    current_arrival_time,
                    current_sc_time_scaled,
                    current_width
                )

                # third, compute and scale profile by spectral energy distribution.
                current_profile *= self.compute_spectrum(
                    self.freqs[current_freq],
                    current_freq_mean,
                    current_freq_width,
                    current_sp_idx,
                    current_sp_run,
                )

                # finally, add to approrpiate slice of model-spectrum matrix.
                model_spectrum[current_freq, :] += current_profile

        return model_spectrum

    def compute_profile(self, 
        times: np.float, 
        arrival_time: np.float, 
        sc_time: np.float,
        width: np.float, 
        ):
        """
        Returns the Gaussian or pulse-broadened temporal profile, depending on input 
        values of width and scattering timescale..
        """

        profile = np.zeros(len(times), dtype=np.float)

        # compute either Gaussian or pulse-broadening function, depending on inputs.
        if (sc_time < 0.05 * width):
            profile = rt.profile.compute_profile_gaussian(times, arrival_time, width)

        else:
            profile = rt.profile.compute_profile_pbf(times, arrival_time, width, sc_time)

        return profile

    def compute_spectrum(self,
        freqs: np.float,
        freq_mean: np.float,
        freq_width: np.float,
        sp_idx: np.float,
        sp_run: np.float,
        ):
        """
        Returns the Gaussian or power-law spectral energy distribution, depending on the 
        desired spectral model (set by self.spectrum_model). 
        """

        spectrum = 0.

        if (self.spectrum_model == "powerlaw"):
            spectrum = rt.spectrum.compute_spectrum_rpl(
                freqs,
                self.reference_freq,
                sp_idx,
                sp_run
            )

        elif (self.spectrum_model == "gaussian"):
            spectrum = rt.spectrum.compute_spectrum_gaussian(freqs, freq_mean, freq_width)

        return spectrum

    def fix_parameter():
        pass

    def get_fit_parameters(self):
        pass

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

        # now, adjust lengths of unset parameters based on input number of components.
        for current_parameter in self.parameters_all:
            if (current_parameter not in model_parameters):
                setattr(self, current_parameter, [None] * self.num_components)
