from fitburst.backend import general
import fitburst.routines as rt
import numpy as np
import sys

class ReaderBaseClass(object):
    """
    A base class for objects that read and pre-process data provided by user.
    """

    def __init__(self):
        """
        Initializes key attributes to be set by all data readers.
        """

        # define basic class attributes here.
        self.burst_parameters = {}
        self.data_full = None
        self.data_weights = None
        self.dedispersion_idx = None
        self.freqs = None
        self.num_freq = None
        self.num_time = None
        self.res_time = None
        self.res_freq = None
        self.times = None

    def dedisperse(self, 
        dm: np.float, 
        arrival_time: np.float,
        reference_freq: np.float = np.inf,
        dm_idx: np.float = general["constants"]["index_dispersion"]
        ):
        """
        Computes a matrix of time-bin indices corresponding to delays from interstellar 
        dispersion of the dynamic spectrum. 
        
        Parameters
        ----------
        dm : np.float
            dispersion measure, in units of pc/cc.

        arrival_time : np.float
            arrival time of the burst, in units of seconds.

        reference_freq : np.float, optional
            reference frequency for supplied arrival time.

        dm_idx : np.float, optional
            exponent of frequency dependence in dispersion delay

        Returns
        -------
        self.dedispersion_delay : np.ndarray
            a matrix of indeces corresponding to incoherent dedispersion of the raw data,
            with dimensions of len(self.freqs) x len(self.times).
        """

        # initialize matrix that contains indices.
        # the following code structure makes use of Numpy broadcasting.
        self.dedispersion_idx = np.zeros(len(self.freqs), dtype=np.int)

        # now loop over dimensions and compute indices.
        delay = arrival_time + rt.ism.compute_time_dm_delay(
            dm,
            general["constants"]["dispersion"],
            dm_idx,
            self.freqs,
            freq2=reference_freq,
        )
        
        # fill initial matrix and transpose.
        self.dedispersion_idx[:] = np.around(
            (delay - self.times[0]) / (self.times[1] - self.times[0]),
        )

    def load_data(self):
        """
        Loads data from file into memory; to be defined by inherited classes.
        """

        pass

    def preprocess_data(
        self, 
        variance_range: list = [0.95, 1.05], 
        variance_weight: np.float = 1.,
        skewness_range: list = [-3., 3.],
        ):
        """
        Applies pre-fit routines for cleaning raw dynamic spectrum (e.g., RFI-masking, 
        baseline subtraction, normalization, etc.).

        Parameters
        ----------
        variance_range : list, optional
            a two-element list containing the range of allowed variances; values outside 
            of this range are flagged as RFI and removed from the data cube.

        variance_weigt : np.float, optional
            a scaling factor applied to variance prior to exision.

        
        skewness_range : list, optional
            a two-element list containing the range of allowed values of skewness; 
            values outside of this range are flagged as RFI and removed from the data cube.

        Returns
        -------
        self.good_freqs : np.ndarray
            an array of boolean values indicating good frequencies.

        Notes
        -----
        This method normalizes and cleans the self.data_full cube.
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

        # finally, downweight zapped channels.
        self.data_full[np.logical_not(good_freq)] = 0.

        # print some info.
        print("INFO: flagged and removed {0} out of {1} channels!".format(
            len(self.freqs) - np.sum(good_freq),
            len(self.freqs),
            )
        )

    def window_data(self, arrival_time: np.float, window: np.float = 0.08):
        """
        Returns a subset of data that is centered and "windowed" on the arrival time.
        
        Parameters
        ----------
        arrival_time : np.float
            arrival time of the burst, in units of seconds.

        window : np.float, optional
            half-range extent of window, in units of seconds.

        Returns
        -------
        data_windowed : np.ndarray
            a matrix containing the dedispersed, windowed spectrum.

        times_windowed : np.ndarray
            an array of timestamps for the dedispersed spectrum.
        """
        
        idx_arrival_time = np.fabs(self.times - arrival_time).argmin()
        num_window_bins = np.around(window / (self.times[1] - self.times[0])).astype(np.int)
        data_windowed = np.zeros((len(self.freqs), num_window_bins * 2), dtype=np.float)

        # compute indeces of min/max window values along time axis.
        for idx_freq in range(len(self.freqs)):
            current_arrival_idx = self.dedispersion_idx[idx_freq]
            data_windowed[idx_freq, :] = self.data_full[
                idx_freq,
                current_arrival_idx - num_window_bins: current_arrival_idx + num_window_bins
            ]
         
        # before finishing, get array of windowed times as well.
        times_windowed = self.times[
            idx_arrival_time - num_window_bins: idx_arrival_time + num_window_bins
        ]

        return data_windowed, times_windowed
