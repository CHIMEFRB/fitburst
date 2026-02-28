#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import numpy as np
import os
import json

# now import some fitburst-specific packages.
from fitburst.utilities import bases


class DataReader(bases.ReaderBaseClass):
    """
    A child class of I/O and processing for generic data stored
    in a .npz or .h5 file, inheriting the basic structure defined in
    ReaderBaseClass().
    """

    def __init__(self, fname):
        # initialise superclass
        super().__init__()

        # ensure file exists, else raise an AssertionError
        self.file_path = fname
        if not os.path.isfile(self.file_path):
            raise IOError(f"Data file not found: {self.file_path}")

        self.file_extension = os.path.splitext(self.file_path)[1]

        # We only need the base class-defined attributes and can update them in-place
        self.model_full = None

    def load_data(self):

        if self.file_extension in [".npz"]:
            self._load_npz()

        elif self.file_extension in [".hdf5", ".h5"]:
            self._load_hdf5()

        else:
            raise IOError(f"Do not recognize file extension: {self.file_extension}")

    def _set_axes(self, metadata):

        self.num_time = metadata["num_time"]
        self.num_freq = metadata["num_freq"]

        self.times_bin0 = metadata["times_bin0"]
        self.res_time = metadata["res_time"]
        self.times = np.arange(self.num_time, dtype=np.float64) * self.res_time + self.times_bin0

        # create frequency channel centre labels from data shape and metadata
        self.freqs_bin0 = metadata["freqs_bin0"]
        self.res_freq = metadata["res_freq"]
        freqs = np.arange(self.num_freq, dtype=np.float64) * self.res_freq + self.freqs_bin0

        self.freqs = freqs

        # store boolean that indicates of input data is already dedispersed or not.
        self.is_dedispersed = metadata["is_dedispersed"]
        # self.dm_incoherent = metadata["dm_incoherent"]

    def _set_burst_parameters(self, burst_parameters):

        # fitburst expects each of these parameters to have values in a list (allows
        # for the possibility of describing multiple components)
        for k, v in burst_parameters.items():
            if not isinstance(v, list):
                self.burst_parameters[k] = [v]
            else:
                self.burst_parameters[k] = v

    def _set_data(self, data_full):
        # derive time information from loaded data.
        nfreq, ntime = data_full.shape
        if nfreq != self.num_freq:
            raise AssertionError(
                "Data shape does not match recorded number of channels"
                f"({nfreq} != {self.num_freq})"
            )
        if ntime != self.num_time:
            raise AssertionError(
                "Data shape does not match recorded number of time samples"
                f"({ntime} != {self.num_time})"
            )
        self.data_full = data_full

    def _set_model(self, model_full):
        # derive time information from loaded data.
        nfreq, ntime = model_full.shape
        if nfreq != self.num_freq:
            raise AssertionError(
                "Model shape does not match recorded number of channels"
                f"({nfreq} != {self.num_freq})"
            )
        if ntime != self.num_time:
            raise AssertionError(
                "Model shape does not match recorded number of time samples"
                f"({ntime} != {self.num_time})"
            )
        self.model_full = model_full

    def _load_npz(self):
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

        # load metadata
        metadata = unpacked_data_set["metadata"].item()
        self._set_axes(metadata)

        # unpack and derive necessary information
        burst_parameters = unpacked_data_set["burst_parameters"].item()
        self._set_burst_parameters(burst_parameters)

        data_full = unpacked_data_set["data_full"]
        self._set_data(data_full)

        # create the weights array, where True = masked
        rfi_mask = metadata["bad_chans"]
        self.good_freq = np.ones(self.num_freq, dtype=bool)
        self.good_freq[rfi_mask] = False

        self.data_weights = (self.good_freq[:, None] & np.ones(self.num_time, dtype=bool)).astype(float)

    def _load_hdf5(self):
        """Load data from the publicly released hdf5 files."""
        import h5py

        with h5py.File(self.file_path, "r") as handler:

            self._set_axes(handler.attrs)

            if "burst_parameters_json" in handler.attrs:
                burst_parameters = json.loads(handler.attrs["burst_parameters_json"])
            else:
                pipeline_parameters = json.loads(handler.attrs["pipeline_parameters_json"])
                par_avail = {"dm": "dm_value", "arrival_time": "arrival_time"}
                burst_parameters = {par_out: pipeline_parameters["dedisperse"][par_in]
                                    for par_out, par_in in par_avail.items()}

            self._set_burst_parameters(burst_parameters)

            data_full = handler["data"][:]
            self._set_data(data_full)

            if "model" in handler:
                model_full = handler["model"][:]
                self._set_model(model_full)

            self.good_freq = handler["good_freq"][:]
            self.data_weights = (handler["flag"][:] & self.good_freq[:, None]).astype(float)

