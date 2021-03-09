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
        expected_subfile_names = ["spectrum", "metadata", "burst_parameters"]
        retrieved_subfile_names = unpacked_data_set.files
        if not all([f in retrieved_subfile_names for f in expected_subfile_names]):
            raise AssertionError(
                f"Data file does not contain one of more of the following keys: "
                f"{expected_subfile_names}"
            )

        metadata = unpacked_data_set["metadata"].item()
        # unpack and derive necessary information
        self.burst_parameters = unpacked_data_set["burst_parameters"].item()
        self.data_full = unpacked_data_set["spectrum"]

        # derive time information from loaded data.
        self.num_freq, self.num_time = self.data_full.shape
        if self.num_freq != metadata["nchan"]:
            raise AssertionError(
                "Data shape does not match recorded number of channels"
                f"({self.num_freq} != {metadata['nchan']})"
            )
        if self.num_time != metadata["ntime"]:
            raise AssertionError(
                "Data shape does not match recorded number of time samples"
                f"({self.num_time} != {metadata['ntime']})"
            )

        # create the weights array, where True = masked
        self.data_weights = np.zeros_like(self.num_freq, dtype=bool)
        rfi_mask = metadata["bad_chans"]
        self.data_weights[rfi_mask] = True

        # create time sample labels from data shape and metadata
        # leave the samples in relative seconds since the beginning of the
        # spectra
        times = np.arange(self.num_time, dtype=np.float64) * metadata["dt"]
        self.times = times
        self.res_time = metadata["dt"]

        # create frequency channel centre labels from data shape and metadata
        freqs = np.arange(self.num_freq, dtype=np.float64) * metadata["chan_bw"]
        freqs += metadata["freq_chan0"]
        # currently have the leading-edge frequency for each channel, add chan_bw / 2
        freqs += abs(metadata["chan_bw"]) / 2.0
        self.freqs = freqs
        self.res_freq = metadata["chan_bw"]
