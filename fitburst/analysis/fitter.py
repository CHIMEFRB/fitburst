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

    def compute_residuals(self, 
        parameter_list: list, 
        times_windowed: np.float, 
        freqs: np.float, 
        data_windowed: np.float
        ):
        """
        Computes the chi-squared statistic used for least-squares fitting.
        """
        
        parameter_dict = self.load_fit_parameters_list(parameter_list)
        self.model.update_parameters(parameter_dict)
        model = self.model.compute_model(times_windowed, freqs)        
        resid = (data_windowed - model)**2
        resid = resid.flatten()

        return resid

    def fit(self, times_windowed, freqs, data_windowed):
        """
        Executing least-squares fitting of the model spectrum to data.
        """

        # convert loaded parameter dictionary/entries into a list for scipy object.
        parameter_list = self.get_fit_parameters_list()

        # do fit!
        lsfit = least_squares(
            self.compute_residuals, 
            parameter_list,
            args = (times_windowed, freqs, data_windowed)
        )

        print(lsfit)

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
        Returns a list of values corresponding to 
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
