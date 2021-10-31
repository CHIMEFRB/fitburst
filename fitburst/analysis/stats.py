from scipy.special import fdtr
import numpy as np
import sys

def compute_test_F(
    chisq_1: float, chisq_2: float, num_fit_parameters_1: int, num_fit_parameters_2: int, 
    num_observations_1: int, num_observations_2: int) -> float:
    """
    Computes the statistical 'F' test value for comparing best-fit fitburst models.

    Parameters
    ----------

    chisq_1 : float
        The best-fit, chi-squared value for fitburst model #1

    chisq_2 : float
        The best-fit, chi-squared value for fitburst model #2

    num_fit_parameters_1 : float
        The number of fitted parameters for fitburst model #1

    num_fit_parameters_2 : float
        The number of fitted parameters for fitburst model #2

    num_observations_1 : float
        The number of data points used for fitburst model #1

    num_observations_2 : float
        The number of data points used for fitburst model #2

    Returns
    -------
    f_test : float
        The F-test (chance coincidence probability) value.
    """

    # compute various terms needed for F-test calc.
    delta_chisq = chisq_2 - chisq_1 
    deg_freedom_1 = num_observations_1 - num_fit_parameters_1
    deg_freedom_2 = num_observations_2 - num_fit_parameters_2
    delta_deg_freedom = deg_freedom_2 - deg_freedom_1
    chisq_reduced_2 = chisq_2 / deg_freedom_2
    probability = 1.

    # now compute the F-test statistic.
    if delta_chisq > 0.:
        f_value = delta_chisq / delta_deg_freedom / chisq_reduced_2
        f_stat = fdtr(delta_deg_freedom, deg_freedom_2, f_value)
        probability -= f_stat

    return probability
