class ReaderBaseClass(object):
    """
    A base class for objects that read and manipulate data provided by user.
    """

    def __init__(self):

        # define class attributes here.
        self.data_full = None
        self.data_windowed = None
        self.freqs = None
        self.times = None

    def dedisperse(self, dm):
        "Returns data that are dedispersed at the supplied DM; to be defined by " + \
        "inherited classes."

        pass

    def load_data(self):
        "Loads data from file into memory; to be defined by inherited classes."

        pass

    def remove_rfi(self):
        "Returns data that are cleaned of RFI; to be defined by inherited classes"

        pass

    def subtract_baseline(self):
        "Returns data off-pulse baselines subtracted to zero RMS; to be defined by " + \
        "inherited classes"

        pass

    def window_data(self, central_time, half_range_time):
        """
        Returns a subset of data that is "windowed" on the central time +/- half range.
        TODO: add feature to window in frequency direction as well.
        """
        
        time_min = central_time - half_range_time
        time_max = central_time + half_range_time

        # compute indeces of min/max window values along time axis.
        pass

