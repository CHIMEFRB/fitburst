"""
Object for Fitting Models against Data of Dynamic Spectra

The LSFitter() object is designed to least-squares fitting of dynamic
spectra based on the model defined and instantiated by the SpectrumModeler()
object. The LSFitter defines several methods to handle the fixing and fitting
of one or more parameters.
"""

from scipy.optimize import least_squares
import fitburst.routines.derivative as deriv
import numpy as np
import sys

class LSFitter:
    """
    A Python object that defines methods and configurations for
    least-squares fitting of radio dynamic spectra.
    """

    def __init__(self, data: float, model: object, good_freq: bool, weighted_fit: bool = True):
        """
        Initializes object with methods and attributes defined in
        the model.SpectrumModeler() class.

        Parameters
        ----------
        model : object
            An instantiation of the SpectrumModeler() object

        good_freq : bool
            An array of boolean values that indicate if a frequency channel is good (True)
            or flagged as radio frequency interference (False)

        weighted_fit : bool, optional
            If set to true, then each channel will be weighted by its standard deviation (or,
            equivalently, the goodness of fit statistic will be a weighted chi-squared value.)

        Returns
        -------
        None : NoneType
            Several atributes are instantiated here
        """

        # load in model into fitter class.
        self.data = data
        self.fit_parameters = []
        self.fit_statistics = {}
        self.model = model

        # initialize fit-parameter list.
        if self.model.scintillation:
            all_parameters = self.model.parameters.copy()

            for current_parameter in self.model.parameters:
                if current_parameter not in ["amplitude", "spectral_index", "spectral_running"]:
                    self.fit_parameters += [current_parameter]

        else:
            self.fit_parameters = self.model.parameters.copy()

        # set parameters for fitter configuration.
        self.good_freq = good_freq
        self.global_parameters = ["dm", "dm_index", "scattering_timescale", "scattering_index"]
        self.success = None
        self.weights = None
        self.weighted_fit = weighted_fit

        # before running fit, determine per-channel weights.
        self._set_weights()

    def compute_hessian(self, data: float, parameter_list: list) -> float:
        """
        Computes the Jacobian matrix for the scipy.optimize.least_squares solver.

        Parameters
        ----------
        parameter_list : list
            A list of names for fit parameters.
        
        Returns
        -------
        jacobian : float
            The Jacobian matrix for the residuals vector supplied to least_squares()

        Notes
        -----
        The parameter_list argument is not actually used in this method, but is 
        supplied in order to conform to the definition of the callable needed by 
        least_squares() for exact calculation of the Jacobian in terms of derivatives.
        """

        # load all parameter values into a dictionary.
        parameter_dict = self.model.get_parameters_dict()

        # before calculating, compute residual.
        residual = data - self.model.compute_model(data = data)

        # define the scale of the Hessian matrix and its labels.
        par_labels_output = []
        par_labels = []
        num_par = 0
        print("here are the fit parameters:", self.fit_parameters)

        for current_par in self.fit_parameters:
            if np.any([current_par == x for x in self.global_parameters]):
                par_labels_output += [current_par]
                par_labels += [current_par]
                num_par += 1

            else:
                par_labels_output += [f"{current_par}{idx+1}" for idx in range(self.model.num_components)]
                par_labels += ([current_par] * self.model.num_components)
                num_par += (self.model.num_components)

        hessian = np.zeros((num_par, num_par), dtype=float)

        # now loop over all fit parameters and compute derivatives.
        for current_par_idx_1, current_par_1 in zip(range(num_par), par_labels):
            current_par_deriv_1 = getattr(deriv, f"deriv_model_wrt_{current_par_1}")
            current_deriv_1 = current_par_deriv_1(
                parameter_dict, self.model, component=(current_par_idx_1 % self.model.num_components)
            )

            # for efficient calculation, only compute one half of the matrix and fill the other half appropriately.
            for current_par_idx_2, current_par_2 in zip(range(current_par_idx_1, num_par), par_labels[current_par_idx_1:]):

                # also compute a given derivative for all burst components.
                current_par_deriv_2 = getattr(deriv, f"deriv_model_wrt_{current_par_2}")
                current_deriv_2 = current_par_deriv_2(
                    parameter_dict, self.model, component=(current_par_idx_2 % self.model.num_components)
                )

                # correct name ordering of mixed partial derivative, if necessary.
                try:
                    current_mixed_deriv = getattr(deriv, f"deriv2_model_wrt_{current_par_1}_{current_par_2}")

                except AttributeError:
                    current_mixed_deriv = getattr(deriv, f"deriv2_model_wrt_{current_par_2}_{current_par_1}")
 
                # only computed mixed derivative for parameters that describe the same component.
                current_deriv_mixed = 0

                if (current_par_idx_1 % self.model.num_components) == (current_par_idx_2 % self.model.num_components):
                    current_deriv_mixed = current_mixed_deriv(
                        parameter_dict, self.model, component=(current_par_idx_2 % self.model.num_components)
                    )

                # finally, compute the hessian here.
                current_hes = 2 * current_deriv_1 * current_deriv_2 - residual * current_deriv_mixed
                hessian[current_par_idx_1, current_par_idx_2] = np.sum(current_hes * self.weights[:, None])
                hessian[current_par_idx_2, current_par_idx_1] = hessian[current_par_idx_1, current_par_idx_2]

        return hessian, par_labels_output

    def compute_jacobian(self, parameter_list: list) -> float:
        """
        Computes the Jacobian matrix for the scipy.optimize.least_squares solver.

        Parameters
        ----------
        parameter_list : list
            A list of names for fit parameters.
        
        Returns
        -------
        jacobian : float
            The Jacobian matrix for the residuals vector supplied to least_squares()

        Notes
        -----
        The parameter_list argument is not actually used in this method, but is 
        supplied in order to conform to the definition of the callable needed by 
        least_squares() for exact calculation of the Jacobian in terms of derivatives.
        """

        # load all parameter values into a dictionary.
        parameter_dict = self.model.get_parameters_dict()

        # define the scale of the Jacobian matrix.
        num_points = len(self.model.times) * len(self.model.freqs)
        jacobian = np.zeros((num_points, len(parameter_list)), dtype=float)
        current_parameter_idx = 0

        # now loop over all fit parameters and compute derivatives.
        for current_parameter in self.fit_parameters:
            current_deriv = getattr(deriv, f"deriv_model_wrt_{current_parameter}")

            # before computing derivatives, account for global parameters.
            num_derivs = self.model.num_components

            if current_parameter in self.global_parameters:
                num_derivs = 1

            # also compute a given derivative for all burst components.
            for current_component in range(num_derivs):
                current_jac = -current_deriv(
                    parameter_dict, self.model, component=current_component
                )
                jacobian[:, current_parameter_idx] = (current_jac * self.weights[:, None]).flat[:]

                # increment parameter index so that jacobian matrix can be filled correctly.
                current_parameter_idx += 1

        return jacobian

    def compute_residuals(self, parameter_list: list) -> float:
        """
        Computes the chi-squared statistic used for least-squares fitting.

        Parameters
        ----------
        parameter_list : list
            A list of names for fit parameters

        times: float
            An array of values corresponding to observing times

        freqs : float
            An array of observing frequencies at which to evaluate spectrum

        spectrum_observed : float
            A matrix of spectrum data, with dimenions that match those of the times
            and freqs arrays

        Returns
        -------
        resid : np.ndarray
            An array of residuals (i.e., the difference between the observed and model spectra)
        """

        # define base model with given parameters.
        parameter_dict = self.load_fit_parameters_list(parameter_list)
        self.model.update_parameters(parameter_dict)
        model = self.model.compute_model(data=self.data)

        # now compute resids and return.
        resid = self.data - model
        resid *= self.weights[:, None]
        resid = resid.flat[:]

        return resid

    def fit(self, exact_jacobian: bool = True) -> None:
        """
        Executes least-squares fitting of the model spectrum to data, and stores
        results to child class.

        Parameters
        ----------
        exact_jacobian : bool, optional
            If set, then use the exact formulation of the Jacobian and supply it 
            as method for the solver to use.

        Returns
        -------
        None
            A dictionary attribute is defined that contains the best-fit results from
            the scipy.optimize.least_aquares solver, as well as estimates of the
            covariance matrix and parameter uncertainties.
        """

        # pylint: disable=broad-except

        # convert loaded parameter dictionary/entries into a list for scipy object.
        parameter_list = self.get_fit_parameters_list()

        # if desired, use exact formulation of Jacobian.
        jac = "2-point"

        if exact_jacobian:
            jac = self.compute_jacobian

        # do fit!
        try:
            results = least_squares(
                self.compute_residuals, 
                parameter_list,
                jac = jac
            )

            self.results = results

            if self.results.success:
                print("INFO: fit successful!")

            else:
                print("INFO: fit didn't work!")

            # now try computing uncertainties and fit statistics.
            self._compute_fit_statistics(self.data, results)

            if self.success:
                print("INFO: derived uncertainties and fit statistics")

        except Exception as exc:
            print("ERROR: solver encountered a failure! Debug!")
            print(sys.exc_info())

    def fix_parameter(self, parameter_list: list) -> None:
        """
        Updates 'fit_parameters' attributes to remove parameters that will
        be held fixed during least-squares fitting.

        Parameters
        ----------
        parameter_list : list
            A list of parameter names that will be fixed to input values during execution
            of the fitting routine

        Returns
        -------
        None : NoneType
            The 'fit_parameters' attribute is updated with supplied parameters removed

        Notes
        -----
        Names of parameters must match those defined in the model.SpectrumModeler class
        """

        print("INFO: removing the following parameters:", ", ".join(parameter_list))

        # removed desired parameters from fit_parameters list.
        for current_parameter in parameter_list:
            self.fit_parameters.remove(current_parameter)

        print("INFO: new list of fit parameters:", ", ".join(self.fit_parameters))

    def get_fit_parameters_list(self) -> list:
        """
        Determines a list of values corresponding to fit parameters.

        Parameters
        ----------
        global_parameters : list, optional
            One or more fit parameters that are tied to all modeled burst components

        Returns
        -------
        parameter_list : list
            A list of floating-point values for fit parameters, which is used by SciPy solver
            as an argument to the residual-computation function
        """

        # pylint: disable=dangerous-default-value

        parameter_list = []

        # loop over all parameters, only extract values for fit parameters.
        for current_parameter in self.model.parameters:
            if current_parameter in self.fit_parameters:
                current_sublist = getattr(self.model, current_parameter)

                if current_parameter in self.global_parameters:
                    parameter_list += [current_sublist[0]]

                else:
                    parameter_list += current_sublist

        return parameter_list

    def load_fit_parameters_list(self, parameter_list: list) -> dict:
        """
        Determines a dictionary where keys are fit-parameter names and values
        are lists (with length self.model.num_components) contain the per-burst
        values of the given parameter/key.

        Parameters
        ----------
        parameter_list : list
            A list of floating-point values for fit parameters, which is used by SciPy solver
            as an argument to the residual-computation function

        global_parameters : list, optional
            One or more fit parameters that are tied to all modeled burst components

        Returns
        -------
        parameter_dict : dict
            A dictionary containing fit parameters as keys and their values as dictionary values
        """

        # pylint: disable=dangerous-default-value

        parameter_dict = {}

        # loop over all parameters, only preserve values for fit parameters.
        current_idx = 0

        for current_parameter in self.model.parameters:
            if current_parameter in self.fit_parameters:

                # if global parameter, load list of length == 1 into dictionary.
                if current_parameter in self.global_parameters:
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
            A matrix of spectrum data, with dimenions that match those of the times
            and freqs arrays

        fit_result : scipy.optimize.OptimizeResult
            The output object from scipy.optimize.least_squares()

        Returns
        -------
        None : None
            The 'fit_statistics' attribute is defined as a Python dicitonary.
        """

        # pylint: disable=broad-except

        # compute various statistics of input data used for fit.
        num_freq, num_time = self.model.num_freq, self.model.num_time
        num_freq_good = int(np.sum(self.good_freq))
        num_fit_parameters = len(fit_result.x)

        self.fit_statistics["num_freq"] = num_freq
        self.fit_statistics["num_freq_good"] = num_freq_good
        self.fit_statistics["num_fit_parameters"] = num_fit_parameters
        self.fit_statistics["num_observations"] = num_freq_good * int(num_time) - num_fit_parameters
        self.fit_statistics["num_time"] = num_time

        # compute chisq values and the fitburst S/N.
        chisq_initial = float(np.sum((self.data * self.weights[:, None])**2))
        chisq_final = float(np.sum(fit_result.fun**2))
        chisq_final_reduced = chisq_final / self.fit_statistics["num_observations"]

        self.fit_statistics["chisq_initial"] = chisq_initial
        self.fit_statistics["chisq_final"] = chisq_final
        self.fit_statistics["chisq_final_reduced"] = chisq_final_reduced
        self.fit_statistics["snr"] = float(np.sqrt(chisq_initial - chisq_final))

        # now compute covarance matrix and parameter uncertainties.
        self.fit_statistics["bestfit_parameters"] = self.load_fit_parameters_list(
            fit_result.x.tolist())
        self.fit_statistics["bestfit_uncertainties"] = None
        self.fit_statistics["bestfit_covariance"] = None

        try:
            hessian = fit_result.jac.T.dot(fit_result.jac)
            covariance = np.linalg.inv(hessian) * chisq_final_reduced
            uncertainties = [float(x) for x in np.sqrt(np.diag(covariance)).tolist()]

            self.covariance = covariance
            self.fit_statistics["bestfit_uncertainties"] = self.load_fit_parameters_list(
                uncertainties)
            self.fit_statistics["bestfit_covariance"] = None # return the full matrix at some point?

        except Exception as exc:
            print(f"ERROR: {exc}; designating fit as unsuccessful...")
            self.success = False

    def _set_weights(self) -> None:
        """
        Sets an attribute containing weights to be applied during least-squares fitting.

        Parameters
        ----------
        spectrum_observed : np.ndarray
            A matrix containing the dynamic spectrum to be analyzed for model fitting.

        Returns
        -------
        None : NoneType
            Two object attributes are defined and used for masking and weighting data during fit.
        """

        # compute RMS deviation for each channel.
        variance = np.mean(self.data**2, axis=1)
        std = np.sqrt(variance)
        bad_freq = np.logical_not(self.good_freq)

        # now compute statistical weights for "good" channels.
        self.weights = np.empty_like(std)

        if self.weighted_fit:
            self.weights[self.good_freq] = 1. / std[self.good_freq]

        else:
            self.weights[self.good_freq] = 1.

        # zero-weight bad channels.
        self.weights[bad_freq] = 0.
