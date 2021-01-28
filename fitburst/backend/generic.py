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
    def __init__(self, fname, data_location="./"):
        # initialise superclass
        super().__init__()

        # ensure file exists, else raise an AssertionError
        self.file_path = f"{data_location}/{fname}"
        if not os.path.isfile(self.file_path):
            raise IOError(f"Data file not found: {self.file_path}")

        # define parameters to be updated by data-retrieval method.
        self.burst_parameters = {}
        self.data_full = None
        self.data_weights = None
        self.data_windowed = None
        self.freqs = None
        self.rfi_freq_count = None
        self.rfi_mask = None
        self.times = None

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
        unpacked_data_set = np.load(self.file_path)

        # ensure required subfiles are present
        expected_subfile_names = ["spectrum", "metadata", "burst_parameters"]
        retrieved_subfile_names = unpacked_data_set.files
        if not all(
                [f in retrieved_subfile_names for f in expected_subfile_names]
        ):
            raise AssertionError(
                f"Data file does not contain one of more of the following "
                f"keys: {expected_subfile_names}"
            )

        # unpack and derive necessary information
        metadata = unpacked_data_set["metadata"]
        self.burst_parameters = unpacked_data_set["burst_parameters"]
        self.data_full = unpacked_data_set["spectrum"]
        self.data_weights = np.ones_like(self.data_full)
        self.rfi_mask = unpacked_data_set["metadata"]["bad_chans"]
        self.rfi_freq_count = len(self.rfi_mask)

        # derive time information from loaded data.
        n_freqs, n_times = self.data_full.shape
        if n_freqs != metadata["nchan"]:
            raise AssertionError(
                "Data shape does not match recorded number of channels"
                f"({n_freqs} != {metadata['nchan']})"
            )
        if n_times != metadata["ntime"]:
            raise AssertionError(
                "Data shape does not match recorded number of time samples"
                f"({n_times} != {metadata['ntime']})"
            )

        # create time sample labels from data shape and metadata
        times = (np.arange(n_times, dtype=np.float64) * metadata["dt"])
        times /= 86400.  # convert to days
        times += metadata["mjd_bin0"]  # absolute offset in MJD
        self.times = times

        # create frequency channel labels from data shape and metadata
        freqs = np.arange(n_freqs, dtype=np.float64) * metadata["chan_bw"]
        freqs += metadata["freq_chan0"]

        self.freqs = freqs  # preserve frequency ordering (lo-hi or hi-lo)
