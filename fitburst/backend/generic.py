#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os

# now import some fitburst-specific packages.
from fitburst.utilities import bases


class DataReader(bases.ReaderBaseClass):
    """
    A child class of I/O and processing for generic data stored
    in a .npz file, inheriting the basic structure defined in
    ReaderBaseClass().
    """

    def __init__(self, fname, data_location="."):
        # initialise superclass
        super().__init__()

        # ensure file exists, else raise an AssertionError
        self.file_path = f"{data_location}/{fname}"
        if not os.path.isfile(self.file_path):
            raise IOError(f"Data file not found: {self.file_path}")

        # We only need the base class-defined attributes and can update them in-place

    def load_data(self):
        """
        Load data from a generic .npz file containing three sub-files:
            spectrum: the raw data in a 2D numpy.ndarray
            metadata: a dictionary containing information required to
                reconstruct data dimensions and physical values, masked
                channels, start MJD, etc.
            burst_parameters: a dictionary containing rough estimates of some
                critical burst parameters that will help the fitters converge
        """
        unpacked_data_set = np.load(self.file_path, allow_pickle=True)

        # ensure required subfiles are present
        expected_subfile_names = ["data_full", "metadata", "burst_parameters"]
        retrieved_subfile_names = unpacked_data_set.files
        if not all([f in retrieved_subfile_names for f in expected_subfile_names]):
            raise AssertionError(
                f"Data file does not contain one of more of the following keys: "
                f"{expected_subfile_names}"
            )

        metadata = unpacked_data_set["metadata"].item()
        # unpack and derive necessary information
        burst_parameters = unpacked_data_set["burst_parameters"].item()
        # fitburst expects each of these parameters to have values in a list (allows
        # for the possibility of describing multiple components)
        for k, v in burst_parameters.items():
            if not isinstance(v, list):
                self.burst_parameters[k] = [v]
            else:
                self.burst_parameters[k] = v

        self.data_full = unpacked_data_set["data_full"]

        # derive time information from loaded data.
        self.num_freq, self.num_time = self.data_full.shape
        if self.num_freq != metadata["num_freq"]:
            raise AssertionError(
                "Data shape does not match recorded number of channels"
                f"({self.num_freq} != {metadata['nchan']})"
            )
        if self.num_time != metadata["num_time"]:
            raise AssertionError(
                "Data shape does not match recorded number of time samples"
                f"({self.num_time} != {metadata['ntime']})"
            )

        # create the weights array, where True = masked
        self.data_weights = np.ones((self.num_freq, self.num_time), dtype=float)
        rfi_mask = metadata["bad_chans"]
        self.data_weights[rfi_mask, :] = 0.

        # create time sample labels from data shape and metadata
        # leave the samples in relative seconds since the beginning of the
        # spectra
        self.res_time = metadata["res_time"]
        times = np.arange(self.num_time, dtype=np.float64) * self.res_time
        self.times = times
        self.times_bin0 = metadata["times_bin0"]

        # create frequency channel centre labels from data shape and metadata
        self.res_freq = metadata["res_freq"]
        freqs = np.arange(self.num_freq, dtype=np.float64) * self.res_freq
        freqs += metadata["freqs_bin0"]
   

        # currently have the leading-edge frequency for each channel, add chan_bw / 2
        freqs += self.res_freq / 2.0
        self.freqs = freqs

        # store boolean that indicates of input data is already dedispersed or not.
        self.is_dedispersed = metadata["is_dedispersed"]
