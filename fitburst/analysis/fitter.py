from scipy.optimize import least_squares
import numpy as np
from . import model

class LSFitter(object):
    """
    A Python object that defines methods and configurations for 
    least-squares fitting of radio dynamic spectra.
    """

    def __init__(self, model_class):
        """
        Initializes object with methods and attributes defined in 
        the model.SpectrumModeler() class.
        """
        
        # load in model into fitter class. 
        self.model = model_class

        # initialize fit-parameter list.
        self.fit_parameters = self.model.parameters_all

    def compute_residuals(self, parameter_list: list, times: np.float, 
        freqs: np.float, data_windowed: np.float):
        """
        Computes the chi-squared statistic used for least-squares fitting.
        """

        # define base model with given parameters.        
        parameter_dict = self.load_fit_parameters_list(parameter_list)
        self.model.update_parameters(parameter_dict)
        model = self.model.compute_model(times, freqs)

        # now compute resids and return.
        resid = data_windowed - model
        resid *= self.weights[:, None]
        resid = resid.flat[:]
        
        # print delta-chisq value.
        print("delta_chisq:", np.sum(data_windowed**2) - np.sum(resid**2))

        return resid

    def fit(self, times, freqs, data_windowed):
        """
        Executing least-squares fitting of the model spectrum to data.
        """

        # convert loaded parameter dictionary/entries into a list for scipy object.
        parameter_list = self.get_fit_parameters_list()
        
        # before running fit, determine per-channel weights.
        self._set_weights(data_windowed)

        # do fit!
        results = least_squares(
            self.compute_residuals, 
            parameter_list,
            args = (times, freqs, data_windowed),
        )

        return results

    def fix_parameter(self, parameter_list: list):
        """
        Updates 'fit_parameters' attributes to remove parameters that will
        be held fixed during least-squares fitting.
        """

        print("INFO: removing the following parameters:", ", ".join(parameter_list))

        # removed desired parameters from fit_parameters list.
        for current_parameter in parameter_list:
            self.fit_parameters.remove(current_parameter)

        print("INFO: new list of fit parameters:", ", ".join(self.fit_parameters))

    def get_fit_parameters_list(self):
        """
        Returns a list of values corresponding to fit parameters.
        """

        parameter_list = []

        # loop over all parameters, only extract values for fit parameters.
        for current_parameter in self.model.parameters_all:
            if (current_parameter in self.fit_parameters):
                parameter_list += getattr(self.model, current_parameter)

        return parameter_list

    def load_fit_parameters_list(self, parameter_list: list):
        """
        Returns a dictionary where keys are fit-parameter names and values 
        are lists (with length self.model.num_components) contain the per-burst 
        values of the given parameter/key.
        """

        parameter_dict = {}

        # loop over all parameters, only preserve values for fit parameters.
        current_idx = 0

        for current_parameter in self.model.parameters_all:
            if (current_parameter in self.fit_parameters):
                parameter_dict[current_parameter] = parameter_list[
                    current_idx:(current_idx + self.model.num_components)
                ]

                current_idx += 1

        return parameter_dict

    def _set_weights(self, input_spectrum):
        """
        Sets an attribute containing weights to be applied during least-squares fitting.
        """

        # compute RMS deviation for each channel.
        variance = np.sum(input_spectrum**2, axis=1)
        std = np.sqrt(variance)
        good_freq = std != 0.
        bad_freq = np.logical_not(good_freq)        

        # now compute statistical weights for "good" channels.
        self.weights = np.empty_like(std)
        self.weights[good_freq] = 1. / std[good_freq]
        self.weights[bad_freq] = 0.
        self.good_freq = good_freq

