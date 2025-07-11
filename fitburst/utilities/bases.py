"""
Base Class for fitburst DataReader Structures

This module contains the base class for the 'DataReader' structure,
which handles all I/O of dynamic spectra. While some methods of the
ReaderBaseClass are common for all data types, other methods (e.g.,
the load_data() method) depend on the nature of the input and are
thus left to be defined by the user for their specific data format.
"""


# standard imports go here.
import sys
import numpy as np

# package specific imports go here.
from fitburst.backend import general
import fitburst.routines as rt

class ReaderBaseClass:
    """
    A base class for objects that read and pre-process data provided by user.
    """

    # pylint: disable=too-many-instance-attributes,too-many-arguments,dangerous-default-value,no-self-use

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
        self.good_freq = None
        self.is_dedispersed = False
        self.num_freq = None
        self.num_time = None
        self.res_time = None
        self.res_freq = None
        self.times = None
        self.times_bin0 = None

    def dedisperse(self, dm_value: float, arrival_time: float, ref_freq: float = np.inf,
                   dm_idx: float = general["constants"]["index_dispersion"],
                   dm_offset: float = None) -> None:
        """
        Computes a matrix of time-bin indices corresponding to delays from
        interstellar dispersion of the dynamic spectrum.

        Parameters
        ----------
        dm_value : float
            The dispersion measure, in units of pc/cc.

        arrival_time : float
            The arrival time of the burst, in units of seconds.

        ref_freq : float, optional
            The reference frequency for supplied arrival time.

        dm_idx : float, optional
            The exponent of frequency dependence in dispersion delay

        dm_offset : float, optional
            The offset in dispersion measure, in units of pc/cc; this option is only used
            if the 'is_dedispersed' attribute is set to True.

        Returns
        -------
        None : NoneType
            This functions has no return value, but instead defines a matrix of indeces
            corresponding to incoherent dedispersion of the raw data, with dimensions
            of (self.num_freq x self.num_time).
        """

        # initialize array that contains indices.
        self.dedispersion_idx = np.zeros(self.num_freq, dtype=int)

        # now compute indicies for a dispersed signal, which will be used
        # in the 'window_data' method for obtain a dedispersed spectrum.
        # NOTE: the dedispersion algorithm is different for dispersed and
        # already-dedispersed spectra, and the following conditionals address this.

        if not self.is_dedispersed:

            # now loop over dimensions and compute indices.
            delay = arrival_time + rt.ism.compute_time_dm_delay(
                dm_value,
                general["constants"]["dispersion"],
                dm_idx,
                self.freqs,
                freq2=ref_freq,
            )

            # fill initial matrix and transpose.
            self.dedispersion_idx[:] = np.ceil(
                (delay - self.times[0]) / (self.res_time),
            )

        elif self.is_dedispersed and dm_offset is not None:

            # now compute delays for initial and adjusted DM.
            delay_1 = rt.ism.compute_time_dm_delay(
                dm_value,
                general["constants"]["dispersion"],
                dm_idx,
                self.freqs,
                freq2=ref_freq,
            )

            delay_2 = rt.ism.compute_time_dm_delay(
                dm_value + dm_offset,
                general["constants"]["dispersion"],
                dm_idx,
                self.freqs,
                freq2=ref_freq,
            )

            # compute discretized delays and finally the difference.
            idx_arrival_time = np.fabs(self.times - arrival_time).argmin()
            dedispersion_idx_1 = np.around((delay_1 - self.times[0]) / (self.res_time))
            dedispersion_idx_2 = np.around((delay_2 - self.times[0]) / (self.res_time))
            self.dedispersion_idx[:] = idx_arrival_time + (dedispersion_idx_2 - dedispersion_idx_1)

    def downsample(self, factor_freq: int = 1, factor_time: int = 1) -> None:
        """
        Downsamples the input spectrum by specified factors across the frequency
        and time axes. The relevant model attributes are updated to downsampled values.
        """

        current_data_full = self.data_full.copy()
        current_freqs = self.freqs.copy()
        current_times = self.times.copy()

        # first, downsample in frequency:
        new_data_full = rt.manipulate.downsample_2d(current_data_full, factor_freq, factor_time)

        # finally, downsample the time and frequency arrays.
        new_freqs = rt.manipulate.downsample_1d(current_freqs, factor_freq)
        new_times = rt.manipulate.downsample_1d(current_times, factor_time)

        # now, replace attributes with downsampled values.
        del self.data_full
        del self.freqs
        del self.num_time
        del self.num_freq
        del self.times
        self.data_full = new_data_full
        self.freqs = new_freqs
        self.times = new_times
        self.num_freq = len(new_freqs)
        self.num_time = len(new_times)
        self.res_freq *= factor_freq
        self.res_time *= factor_time

        # if the good-frequency array is defined, compute the downsampled version as well.
        if self.good_freq is not None:
            current_good_freq = self.good_freq.copy()
            new_good_freq = rt.manipulate.downsample_1d(current_good_freq,
                                                        factor_freq, boolean=True)

            # replace attribute.
            del self.good_freq
            self.good_freq = new_good_freq

            # ensure that previously-labeled bad frequencies remain that way if downsampled.
            self.data_full[np.logical_not(self.good_freq)] = 0.

        # if the good-frequency array is defined, compute the downsampled version as well.
        if self.data_weights is not None:
            current_data_weights = self.data_weights.copy()
            new_data_weights = rt.manipulate.downsample_2d(current_data_weights,
                                                           factor_freq, factor_time)

            # replace attribute.
            del self.data_weights
            self.data_weights = new_data_weights

    def load_data(self) -> None:
        """
        Loads data from file into memory; to be defined by inheriting DataReader
        classes in backends/ subdirectory.

        Notes
        -----
        This method must define the following:
            self.data_full
            self.data_weights
            self.freqs
            self.times
            self.num_freq
            self.num_time
        """

        sys.exit("ERROR: load_data() must be defined for input data format!")

    def preprocess_data(self, apply_cut_variance: bool = False, apply_cut_skewness: bool = False, 
                        normalize_variance: bool = True, remove_baseline: bool = False, 
                        skewness_range: list = [-3., 3.], variance_range: list = [0.2, 0.8], 
                        variance_weight: float = 1.) -> None:
        """
        Applies pre-fit routines for cleaning raw dynamic spectrum (e.g., RFI-masking,
        baseline subtraction, normalization, etc.).

        Parameters
        ----------
        apply_cut_variance : bool, optional
            if True, then update mask to exclude channels with variance values that exceed 
            the range specified in the 'variance_range' list

        apply_cut_skewness : bool, optional
            if True, then update mask to exclude channels with skewness values that exceed 
            the range specified in the 'skewness_range' list

        normalize_variance: bool, optional
            if true, then normalize variances relative to the largest value.

        skewness_range : list, optional
            a two-element list containing the range of allowed values of skewness;
            values outside of this range are flagged as RFI and removed from the data cube.

        remove_baseline : bool, optional
            if True, then renormalize data and remove baseline

        variance_range : list, optional
            a two-element list containing the range of allowed variances; values outside
            of this range are flagged as RFI and removed from the data cube.

        variance_weigt : np.float, optional
            a scaling factor applied to variance prior to exision.

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

        # just to be sure, loop over data and ensure channels aren't "bad".
        for idx_freq in range(self.num_freq):
            if good_freq[idx_freq]:
                if self.data_full[idx_freq, :].min() == self.data_full[idx_freq, :].max():
                    print(f"ERROR: bad data value of {self.data_full[idx_freq, :].min()} in channel {idx_freq}!")
                    good_freq[idx_freq] = False

        # if desired, normalize data and remove baseline.
        mean_spectrum = np.sum(self.data_full * self.data_weights, -1)
        #good_freq[np.where(mean_spectrum == 0.)] = False
        mean_spectrum[good_freq] /= mask_freq[good_freq]

        if remove_baseline:
            self.data_full[good_freq] /= mean_spectrum[good_freq][:, None]
            self.data_full[good_freq] -= 1
            self.data_full[np.logical_not(self.data_weights)] = 0

        # compute variance and skewness of data.
        variance = np.sum(self.data_full ** 2, -1)
        variance[good_freq] /= mask_freq[good_freq]
        skewness = np.sum(self.data_full ** 3, -1)
        skewness[good_freq] /= mask_freq[good_freq]
        skewness_mean = np.mean(skewness[good_freq])
        skewness_std = np.std(skewness[good_freq])

        # if desired, normalize variance relative to maximum value.

        if normalize_variance:
            variance[good_freq] /= np.max(variance[good_freq])

        # now update good-frequencies list based on variance/skewness thresholds.
        if apply_cut_variance: 
            good_freq = np.logical_and(good_freq, (variance / variance_weight < variance_range[1]))
            good_freq = np.logical_and(good_freq, (variance / variance_weight > variance_range[0]))

        if apply_cut_skewness:
            good_freq = np.logical_and(good_freq,
                                       (skewness < skewness_mean + skewness_range[1] * skewness_std))
            good_freq = np.logical_and(good_freq,
                                       (skewness > skewness_mean + skewness_range[0] * skewness_std))

        # finally, downweight zapped channels.
        self.data_full[np.logical_not(good_freq)] = 0.

        # stash good frequencies and print some info.
        self.good_freq = good_freq
        num_freq_bad = self.num_freq - np.sum(self.good_freq)

        print(f"INFO: flagged and removed {num_freq_bad} out of {self.num_freq} channels!")

    def window_data(self, arrival_time: float, window: float = 0.08) -> tuple:
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
        num_window_bins = np.around(window / self.res_time).astype(int)
        data_windowed = np.zeros((self.num_freq, num_window_bins * 2), dtype=float)

        # compute indeces of min/max window values along time axis.
        for idx_freq in range(self.num_freq):
            current_arrival_idx = self.dedispersion_idx[idx_freq]
            data_windowed[idx_freq, :] = self.data_full[idx_freq,
                current_arrival_idx - num_window_bins: current_arrival_idx + num_window_bins]

        # before finishing, get array of windowed times as well.
        times_windowed = self.times[
            idx_arrival_time - num_window_bins: idx_arrival_time + num_window_bins
        ]

        return data_windowed, times_windowed
