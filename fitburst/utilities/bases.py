import fitburst.routines as rt
import numpy as np
import sys

class ReaderBaseClass(object):
    """
    A base class for objects that read and manipulate data provided by user.
    """

    def __init__(self):

        # define class attributes here.
        self.data_full = None
        self.data_weights = None
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

    def preprocess_data(
        self, 
        variance_range: list = [0.95, 1.05], 
        variance_weight: np.float = 1.,
        skewness_range: list = [-3., 3.],
        ):
        """
        Applies pre-fit routines for cleaning raw dynamic spectrum (e.g., RFI-masking, 
        baseline subtraction, normalization, etc.)
        """

        # define weight and mask arrays.        
        mask_freq = np.sum(self.data_weights, -1)
        good_freq = mask_freq != 0

        # normalize data and remove baseline.
        mean_spectrum = np.sum(self.data_full * self.data_weights, -1)
        mean_spectrum[good_freq] /= mask_freq[good_freq]
        self.data_full[good_freq] /= mean_spectrum[good_freq][:, None]
        self.data_full[good_freq] -= 1
        self.data_full[np.logical_not(self.data_weights)] = 0

        # compute variance and skewness of data.
        variance = np.sum(self.data_full**2, -1) 
        variance[good_freq] /= mask_freq[good_freq]
        skewness = np.sum(self.data_full**3, -1) 
        skewness[good_freq] /= mask_freq[good_freq]
        skewness_mean = np.mean(skewness[good_freq])
        skewness_std = np.std(skewness[good_freq])

        # now update good-frequencies list based on variance/skewness thresholds.
        good_freq = np.logical_and(
            good_freq, 
            (variance / variance_weight < variance_range[1])
        )
        good_freq = np.logical_and(
            good_freq, 
            (variance / variance_weight > variance_range[0])
        )
        good_freq = np.logical_and(
            good_freq, 
            (skewness < skewness_mean + skewness_range[1] * skewness_std)
        )
        good_freq = np.logical_and(
            good_freq, 
            (skewness > skewness_mean + skewness_range[0] * skewness_std)
        )

        # print some info.
        print("INFO: flagged and removed {0} out of {1} channels!".format(
            len(self.freqs) - np.sum(good_freq),
            len(self.freqs),
            )
        )

    def window_data(self, central_time, half_range_time):
        """
        Returns a subset of data that is "windowed" on the central time +/- half range.
        TODO: add feature to window in frequency direction as well.
        """
        
        time_min = central_time - half_range_time
        time_max = central_time + half_range_time

        # compute indeces of min/max window values along time axis.
        pass

