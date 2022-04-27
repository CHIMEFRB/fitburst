from scipy.optimize import least_squares, curve_fit
from fitburst.backend.baseband import *
import numpy as np
import matplotlib.pyplot as plt
from . import model
from . import fitter_plotting

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
        self.fit_parameters = self.model.parameters.copy()

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

            if self.success:
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
        for current_parameter in self.model.parameters:
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

        for current_parameter in self.model.parameters:
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
            print(f"ERROR: {exc}; designating fit as unsuccessful...")
            self.success = False

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

def fit_LS(profile : np.ndarray, xvals : np.ndarray, peaks : np.ndarray,
    event_id : str = '', model : str = 'emg', res : float = 1., ICs : np.ndarray = None,
    tight : float = None, bounds : dict = {}, diagnostic_plots : bool = True) -> tuple:
    """
    Least squares fitting using curve_fit.
    
    Parameters
    ----------
    profile : np.ndarray
        Pulse profile to be fit
    x_vals : np.ndarray
        Timestamp of every time bin (same size as profile)
    peaks : np.ndarray, optional
        Array with the peak times of every burst component in s
    event_id : str, optional
        Event ID of the FRB
    model : str, optional
        For now, the only option is 'emg'
    res : float, optional
        Resolution (either time in s or freq in MHz)
    ICs : dict, optional
        You can provide a dict with ICs if the ones calculated by the code are not good enough.
        Every key of the dict is a parameter i.e. for EMG: ["A", "mu", "sigma", "lam"]
        To every key, provide an array with length = npeaks with the initial param guesses.
        Note: This feature doesn't work currently.
    tight : float
        If provided, narrows the parameter bounds (for all parameters) provided to curve_fit
        The provided float is the upper bound for the parameter sigma, especially useful to force curve_fit
        to fit narrower peaks.
    bounds : dict, optional
        You can provide a dict with upper bounds if the ones calculated by the code are too wide.
        Every key of the dict is a parameter i.e. for EMG: ["A", "mu", "sigma", "lam"]
        To every key, provide an array with length = npeaks with the upper bound for the LS fit 
        for each param of each burst component.
    
    Returns
    -------
    tuple
        LS fit to data, LS best fit params, LS best fit 1 sigma error on params, full covariance matrix.
    
    """
    bounds = {}
    if ICs is None and model == 'emg':
        ICs = []
        peaks = np.array(sorted(peaks))
        ub = []
        lb = []
        keys = ["A", "mu", "sigma", "lam"]
        for k in keys:
            try:
                bounds[k]
            except KeyError:
                    bounds[k] = np.inf
            if type(bounds[k]) != np.ndarray and k != "lam":
                bounds[k] = np.zeros(len(peaks)) + bounds[k]
        if tight is not None:
            mu_max = peaks
            mu_min = peaks - np.diff(np.append(peaks, max(xvals)))/3
        else:
            mu_max = peaks
            mu_min = peaks - np.diff(np.append(peaks, max(xvals)))/2
            print(mu_min)
            for i in range(len(peaks)):
                if i > 0:
                    mu_min[i] = max(peaks[i-1], mu_min[i])
            print(mu_min)
        mu_min[mu_min < 0] = 0
        for i in range(len(peaks)):
            diff = abs(xvals - peaks[i])
            lhs = profile[0:np.where(profile == max(profile))[0][0]]
            if tight is not None:
                ub.extend([bounds["A"][i],mu_max[i], tight])
                sigma_guess = tight*0.9
            else:
                ub.extend([bounds["A"][i],mu_max[i], bounds["sigma"][i]])
                sigma_guess = res*len(lhs[lhs > max(profile)/2])
            ICs.extend([max(profile)*sigma_guess, peaks[i], sigma_guess])
            lb.extend([0,mu_min[i], 0])
        #scattering tail is the same for all peaks
        ICs.extend([(2*sigma_guess)])
        ub.extend([bounds["lam"]])
        lb.extend([0])
        print(lb, ICs, ub)
        f = sum_emg 
    elif ICs is None and model == 'rpl':
        f = rpl
        ICs = [max(profile),max(profile), 0]
        ub = [np.inf, np.inf,np.inf]
        lb = [0, -np.inf, -np.inf]
        print(lb, ICs, ub)
    else:
        ub, lb = [], []
        for i in range(len(peaks)):
            ub.extend([np.inf,peaks[i] + 150, max(xvals)])
            lb.extend([0,peaks[i] - 150, 0])
        ub.extend([np.inf])
        lb.extend([0])
        print(lb, ICs, ub)
        f = sum_emg
    x = xvals
    y = profile
    plt.plot(x,y)
    for p in peaks:
        plt.axvline(p, color = 'r')
    plt.plot(x,f(x,ICs),color = 'orange')
    plt.show()
    popt,pcov = curve_fit(f,x, y, p0=ICs, bounds = (lb, ub),maxfev = 100000000)
    if diagnostic_plots:
        print('Curve_fit params: ' + str(popt))
        fitter_plotting.show_fit(profile, xvals, popt, res, event_id, m = model)
    
    return f(xvals, popt), popt, np.sqrt(np.diag(pcov)), pcov
