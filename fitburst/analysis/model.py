class Parameters(object):
    """
    A python structure that contains all information regarding parameters that 
    describe and are used to compute models of dynamic spectra.
    """

    def __init__(self):
        # set defaults here.
        pass

    def set_dms(self):
        # overload DM value(s).
        pass

    def set_n_components(self):
        # overload number of burst components.
        pass

    def set_sc_times(self):
        # overload scattering timescale(s).
        pass

    def set_spectra(self):
        # overload spectral parameters.
        pass

    def set_toas(self):
        # overload times of arrival (TOAs).
        pass

    def set_widths(self):
        # overload temporal width(s).
        pass

    def summary(self):
        # print a summary of parameter collection.
        pass
