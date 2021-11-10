from scipy.optimize import least_squares
import numpy as np
from . import model

class LSFitter(object):
    """
    A Python object that defines methods and configurations for 
    least-squares fitting of radio dynamic spectra.
    """

    def __init__(self, model_class: object):
        """
        Initializes object with methods and attributes defined in 
        the model.SpectrumModeler() class.
        """
        
        # load in model into fitter class.
        self.fit_statistics = {}
        self.model = model_class

        # initialize fit-parameter list.
        self.fit_parameters = self.model.parameters_all.copy()

        # set parameters for fitter configuration.
        self.weighted_fit = True
        self.success = None

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

            self.success = results.success

            if self.success:
                print("INFO: fit successful!")

            else:
                print("INFO: fit didn't work!")

            # now try computing uncertainties and fit statistics.
            self._compute_fit_statistics(spectrum_observed, results)
            print("INFO: derived uncertainties and fit statistics")

        except Exception as exc:
            print("ERROR: solver encountered a failure! Debug!")
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

    def _compute_fit_statistics(self, spectrum_observed: float, fit_result: object) -> None:
        """
        Computes and stores a variety of statistics and best-fit results. 

        Parameters
        ----------
        spectrum_observed : np.ndarray
            a matrix of spectrum data, with dimenions that match those of the times 
            and freqs arrays

        fit_result : scipy.optimize.OptimizeResult
            the output object from scipy.optimize.least_squares()

        Returns
        -------
        None : None
            The 'fit_statistics' attribute is defined as a Python dicitonary.
        """

        # compute various statistics of input data used for fit.
        num_freq, num_time = spectrum_observed.shape
        num_freq_good = int(np.sum(self.good_freq))
        num_fit_parameters = len(fit_result.x)

        self.fit_statistics["num_freq_good"] = num_freq_good
        self.fit_statistics["num_fit_parameters"] = num_fit_parameters
        self.fit_statistics["num_observations"] = num_freq_good * int(num_time) - num_fit_parameters
        self.fit_statistics["num_time"] = num_time

        # compute chisq values and the fitburst S/N.
        chisq_initial = np.sum((spectrum_observed * self.weights[:, None])**2)
        chisq_final = np.sum(fit_result.fun**2)
        chisq_final_reduced = chisq_final / self.fit_statistics["num_observations"]

        self.fit_statistics["chisq_initial"] = chisq_initial
        self.fit_statistics["chisq_final"] = chisq_final
        self.fit_statistics["chisq_final_reduced"] = chisq_final_reduced
        self.fit_statistics["snr"] = np.sqrt(chisq_initial - chisq_final)

        # now compute covarance matrix and parameter uncertainties.       
        self.fit_statistics["bestfit_parameters"] = self.load_fit_parameters_list(fit_result.x.tolist())
        self.fit_statistics["bestfit_uncertainties"] = None
        self.fit_statistics["bestfit_covariance"] = None

        try:
            hessian = fit_result.jac.T.dot(fit_result.jac)
            covariance = np.linalg.inv(hessian) * chisq_final_reduced
            uncertainties = np.sqrt(np.diag(covariance)).tolist()
 
            self.fit_statistics["bestfit_uncertainties"] = self.load_fit_parameters_list(uncertainties)
            self.fit_statistics["bestfit_covariance"] = None # return the full matrix at some point?

        except Exception as exc:
            print(exc)

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
        variance = np.mean(spectrum_observed**2, axis=1)
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

