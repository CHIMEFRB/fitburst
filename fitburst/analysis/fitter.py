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

        # set parameters for fitter configuration.
        self.weighted_fit = True

    def compute_residuals(self, parameter_list: list, times: float, 
        freqs: float, spectrum_observed: float) -> float:
        """
        Computes the chi-squared statistic used for least-squares fitting.

        Parameters
        ----------
        parameter_list : list
            a list of names for fit parameters

        times: np.ndarray
             an array of values corresponding to observing times
           
        freqs : np.ndarray
            an array of observing frequencies at which to evaluate spectrum

        spectrum_observed : np.ndarray
            a matrix of spectrum data, with dimenions that match those of the times 
            and freqs arrays

        Returns
        -------
        resid : np.ndarray
            an array of residuals (i.e., the difference between the observed and model spectra)
        """

        # define base model with given parameters.        
        parameter_dict = self.load_fit_parameters_list(parameter_list)
        self.model.update_parameters(parameter_dict)
        model = self.model.compute_model(times, freqs)

        # now compute resids and return.
        resid = spectrum_observed - model
        resid *= self.weights[:, None]
        resid = resid.flat[:]
        
        return resid

    def fit(self, times: float, freqs: float, spectrum_observed: float) -> None:
        """
        Executes least-squares fitting of the model spectrum to data, and stores 
        results to child class.

        Parameters
        ----------
        times: np.ndarray
             an array of values corresponding to observing times
           
        freqs : np.ndarray
            an array of observing frequencies at which to evaluate spectrum

        spectrum_observed : np.ndarray
            a matrix of spectrum data, with dimenions that match those of the times 
            and freqs arrays

        Returns
        -------
        None
            a dictionary attribute is defined that contains the best-fit results from 
            the scipy.optimize.least_aquares solver, as well as estimates of the 
            covariance matrix and parameter uncertainties.
        """

        # convert loaded parameter dictionary/entries into a list for scipy object.
        parameter_list = self.get_fit_parameters_list()
        
        # before running fit, determine per-channel weights.
        self._set_weights(spectrum_observed)

        # do fit!
        try:
            results = least_squares(
                self.compute_residuals, 
                parameter_list,
                args = (times, freqs, spectrum_observed)
            )

            print("INFO: fit successful!")

            # store resulting object to class.
            self.bestfit_results = {}
            self.bestfit_results["solver"] = results
            self.bestfit_results["parameters"] = results.x

        except Exception as exc:
            print("ERROR: solver encountered a failure! Debug!")
            print(exc)

        # now try computing uncertainties from numerical Jacobian.
        try:
            chisq_reduced = np.sum(results.fun**2) / (len(results.fun) - len(results.x))
            uncertainties, covariance = self._compute_uncertainties(results.jac, chisq_reduced)
            self.bestfit_results["covariance"] = covariance
            self.bestfit_results["uncertainties"] = uncertainties
            print("INFO: derived uncertainties from Jacobian data")

        except Exception as exc:
            print("ERROR: could not compute uncertainties!")
            print(exc)

    def fix_parameter(self, parameter_list: list) -> None:
        """
        Updates 'fit_parameters' attributes to remove parameters that will
        be held fixed during least-squares fitting.

        Parameters
        ----------
        parameter_list : list
            a list of parameter names that will be fixed to input values during execution 
            of the fitting routine.

        Returns
        -------
        None
            the 'fit_parameters' attribute is updated with supplied parameters removed.

        Notes
        -----
        Names of parameters must match those defined in the model.SpectrumModeler class.
        """

        print("INFO: removing the following parameters:", ", ".join(parameter_list))

        # removed desired parameters from fit_parameters list.
        for current_parameter in parameter_list:
            self.fit_parameters.remove(current_parameter)

        print("INFO: new list of fit parameters:", ", ".join(self.fit_parameters))

    def get_fit_parameters_list(self, global_parameters: list = ["dm", "scattering_timescale"]) -> list:
        """
        Determines a list of values corresponding to fit parameters.

        Parameters
        ----------
        global_parameters : list, optional
            one or more fit parameters that are tied to all modeled burst components. 

        Returns
        -------
        parameter_list : list
            a list of floating-point values for fit parameters, which is used by SciPy solver 
            as an argument to the residual-computation function.

        """

        parameter_list = []

        # loop over all parameters, only extract values for fit parameters.
        for current_parameter in self.model.parameters_all:
            if current_parameter in self.fit_parameters:
                current_sublist = getattr(self.model, current_parameter)

                if current_parameter in global_parameters:
                    parameter_list += [current_sublist[0]]

                else:
                    parameter_list += current_sublist

        return parameter_list

    def load_fit_parameters_list(self, parameter_list: list, 
        global_parameters: list = ["dm", "scattering_timescale"]) -> dict:
        """
        Determines a dictionary where keys are fit-parameter names and values 
        are lists (with length self.model.num_components) contain the per-burst 
        values of the given parameter/key.

        Parameters
        ----------
        parameter_list : list
            a list of floating-point values for fit parameters, which is used by SciPy solver 
            as an argument to the residual-computation function.

        global_parameters : list, optional
            one or more fit parameters that are tied to all modeled burst components. 

        Returns
        -------
        parameter_dict : dict
            a dictionary containing fit parameters as keys and their values as dictionary values.
        """

        parameter_dict = {}

        # loop over all parameters, only preserve values for fit parameters.
        current_idx = 0

        for current_parameter in self.model.parameters_all:
            if (current_parameter in self.fit_parameters):
                # if global parameter, load list of length == 1 into dictionary.
                if current_parameter in global_parameters:
                    parameter_dict[current_parameter] = [parameter_list[current_idx]] 
                    current_idx += 1

                else:
                    parameter_dict[current_parameter] = parameter_list[
                        current_idx : (current_idx + self.model.num_components)
                    ]

                    current_idx += self.model.num_components


        return parameter_dict

    def _compute_uncertainties(self, jacobian: float, residual_variance: float) -> float:
        """
        Computes statistical uncertainties from output of least-squares solver.

        Parameters
        ----------
        jacobian : np.ndarray
            the Jacobian matrix computed from the scipy.optimize.least_squares fitting routine

        residual_variance : float
            the normalization factor for computing the covariance matrix and uncertainties.

        Returns
        -------
        uncertainties : np.ndarray
            array of uncertainties for fit parameters

        covariance : np.ndarray
            covariance matrix for fit parameters
        """

        # compute hessian and covariance (inverse-hessian) matrices 
        # from input jacobian at best-fit location in phase space.
        hessian = jacobian.T.dot(jacobian)
        covariance = np.linalg.inv(hessian) * residual_variance
        
        # finally, compute uncertainties from diagonal elements.
        uncertainties = np.sqrt(np.diag(covariance))

        return uncertainties, covariance

    def _set_weights(self, spectrum_observed: float) -> None:
        """
        Sets an attribute containing weights to be applied during least-squares fitting.

        Parameters
        ----------
        spectrum_observed : np.ndarray
            matrix containing the dynamic spectrum to be analyzed for model fitting.

        Returns
        -------
        None
            two object attributes are defined and used for masking and weighting data during fit.
        """

        # compute RMS deviation for each channel.
        variance = np.sum(spectrum_observed**2, axis=1)
        std = np.sqrt(variance)
        good_freq = std != 0.
        bad_freq = np.logical_not(good_freq)        

        # now compute statistical weights for "good" channels.
        self.weights = np.empty_like(std)

        if self.weighted_fit:
            self.weights[good_freq] = 1. / std[good_freq]

        else:
            self.weights[good_freq] = 1.

        self.weights[bad_freq] = 0.
        self.good_freq = good_freq

