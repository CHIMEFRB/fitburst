import scipy as sp
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

        super().__init__(self, model_class)
