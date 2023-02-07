import numpy as np
import datetime
import requests
import glob
import pytz
import sys

# now import some fitburst-specific packages.
from fitburst.utilities import bases
from . import telescopes


class DataReader(bases.ReaderBaseClass):
    """
    A child class of I/O and processing for CHIME/FRB data, inheriting the basic
    structure defined in ReaderBaseClass().
    """

    def __init__(self, eventid, beam_id: int = 0, data_location: str = "/data/frb-archiver"):

        # before anything else, initialize superclass.
        super().__init__()

        # now, ensure eventid makes sense before retrieving data.
        self.eventid = eventid
        assert isinstance(self.eventid, int)
        print("CHIMEFRBReader executed:")

        # define CHIME/FRB-specific parameters to be updated by data-retrieval method.
        self.beam_id = beam_id
        self.downsample_factor = None
        self.files = []
        self.fpga_count_start = None
        self.fpga_count_total = None
        self.fpga_frame0_nano = None
        self.frbmaster_request_status = None
        self.rfi_freq_count = None
        self.rfi_mask = None

        # as a default, grab data from FRBMaster.
        print("... grabbing metadata for eventID {0}".format(self.eventid))
        self._retrieve_metadata_frbmaster(self.eventid, beam_id=self.beam_id)

    def get_parameters(self, pipeline: str = "L1") -> dict:
        """
        Returns a dictionary containing parameters as keys and their FRBMaster entries
        stored as values.

        Parameters
        ----------
        pipeline: str, optional
            The name of CHIME/FRB pipeline for which to grab locked parameters.
            Current options are: L1, dm, fitburst.

        Returns
        -------
        parameter_dict : dict
            A python dicitonary containing parameters of dynamic spectra, with available 
            pipeline values replacin fitburst default values.
        """

        parameter_dict = {}

        ### if fitburst results exist and are desired, grab those.
        if bool(self.burst_parameters["fitburst"]) and pipeline == "fitburst":
            current_round = "round_2"

            if "scattering_timescale" in self.burst_parameters["fitburst"]["round_3"]:
                current_round = "round_3"

            for current_key in self.burst_parameters["fitburst"][current_round].keys():
                parameter_dict[current_key] = self.burst_parameters["fitburst"][
                    current_round
                ][current_key]

            # adjust certain FRBMaster entries if burst has multiple components.
            num_components = len(parameter_dict["arrival_time"])
            parameter_dict["ref_freq"] = parameter_dict["ref_freq"] * num_components

            # add parameters that are not reported in FRBMaster here.
            parameter_dict["dm_index"] = [-2.0] * num_components
            parameter_dict["scattering_index"] = [-4.0] * num_components
            

        ### if instead the DM-pipeline results exist and are desired, grab those.
        elif bool(self.burst_parameters["dm-pipeline"]) and pipeline == "dm":
            print("woohoo DM pipleine")
            parameter_dict["amplitude"] = [-3.0]
            parameter_dict["burst_width"] = self.burst_parameters["dm-pipeline"]["width"]
            parameter_dict["dm"] = self.burst_parameters["dm-pipeline"]["dm"]
            parameter_dict["dm_index"] = [-2.0]
            parameter_dict["ref_freq"] = [telescopes["chimefrb"]["pivot_freq"]["spectrum"]]
            parameter_dict["scattering_index"] = [-4.0]
            parameter_dict["scattering_timescale"] = [0.0]
            parameter_dict["spectral_index"] = [-1.0]
            parameter_dict["spectral_running"] = [0.0]

            if self.fpga_frame0_nano is not None:
                parameter_dict["arrival_time"] = [
                    pytz.utc.localize(self.burst_parameters["dm-pipeline"]["timestamp_utc"][0]).timestamp() - \
                    (self.fpga_frame0_nano * 1e-9)
                ]

        ### if the default mode is chosen, just grab the parameters determined by L1.
        elif bool(self.burst_parameters["L1"]) and pipeline == "L1":
            print("ok at least there is L1")

            # L1 only estimates parameters for one component, so just create a dictionary 
            # corresponding to one burst component. Use guesses for values not estimated by L1.
            parameter_dict["amplitude"] = [-3.0]
            parameter_dict["arrival_time"] = [
                self.burst_parameters["L1"]["timestamp_fpga"] * 
                telescopes["chimefrb"]["fpga"]["time_per_sample"] 
            ]
            parameter_dict["burst_width"] = [0.05]
            parameter_dict["dm"] = [self.burst_parameters["L1"]["dm"]]
            parameter_dict["dm_index"] = [-2.0]
            parameter_dict["ref_freq"] = [telescopes["chimefrb"]["pivot_freq"]["spectrum"]]
            parameter_dict["scattering_index"] = [-4.0]
            parameter_dict["scattering_timescale"] = [0.0]
            parameter_dict["spectral_index"] = [-1.0]
            parameter_dict["spectral_running"] = [0.0]
            

        else:
            print("ERROR: no parameters retrieved from FRBMaster!")

        ### return parameter dictionary.
        return parameter_dict

    def load_data(self, files: list) -> None:
        """
        Load data from CHIME/FRB msgpack data files.

        Parameters
        ----------
        files: list
            A list of msgpack files to load

        """

        try:
            from cfod.chime_intensity import unpack_datafiles

        except ImportError as err:
            print("Unable to import from cfod")
            print("Please ensure this package is installed.")
            print(err)

        unpacked_data_set = unpack_datafiles(files)
        self.data_full = unpacked_data_set[0]
        self.data_weights = unpacked_data_set[1]
        self.fpga_count_start = unpacked_data_set[2]
        self.fpga_count_total = unpacked_data_set[3]
        self.downsample_factor = unpacked_data_set[4]
        self.rfi_mask = unpacked_data_set[5]
        self.fpga_frame0_nano = (unpacked_data_set[6])[0]

        # derive time information from loaded data.
        n_freqs, n_times = self.data_full.shape
        times = np.arange(n_times, dtype=np.int64) + (self.downsample_factor // 2)
        times *= (
            telescopes["chimefrb"]["num_frames_per_sample"]
            * telescopes["chimefrb"]["num_factor_upchannel"]
        )
        times += self.fpga_count_start[0]
        self.times = times * telescopes["chimefrb"]["fpga"]["time_per_sample"]

        # now derive frequency information.
        freqs = np.arange(n_freqs, dtype=np.float64)
        freqs *= -(
            telescopes["chimefrb"]["bandwidth"] / telescopes["chimefrb"]["num_channels"]
        )
        freqs += (
            telescopes["chimefrb"]["fpga"]["freq_top"]
            + telescopes["chimefrb"]["bandwidth"]
            / telescopes["chimefrb"]["fpga"]["num_channels"]
            / 2
        )
        self.freqs = freqs[::-1]

        # define index values before exiting.
        self.num_freq = len(self.freqs)
        self.num_time = len(self.times)
        self.res_freq = self.freqs[1] - self.freqs[0]
        self.res_time = self.times[1] - self.times[0]

    def _retrieve_metadata_frbmaster(
        self, eventid: str, beam_id: int = 0, mountpoint: str = "/data/chime"
    ) -> None:
        """
        This internal methods executes CHIME/FRB-specific actions for retrieving the necessary 
        metadata from the FRBMaster database, and requires direct network access to raw intensity 
        data (i.e., should be run at the CHIME or CANFAR sites).

        Parameters
        ----------
        eventid : str
            The CHIME/FRB ID for the event of interest.

        beam_id : int, optional
            The index of a list of recorded beams, corresponding to the desired data set.
            (This list is ordered in decreasing S/N; beam_id = 0 corresponds to the highest-S/N beam.) 

        mountpoint : str, optional
            The local root directory where raw intensity data are stored.

        Returns
        -------
        None : None
            This method sets a large of number of attributes that comprise the DataReader object.
        """

        try:
            from cfod.chime_intensity import natural_keys
            from chime_frb_api.backends.frb_master import FRBMaster

        except ImportError as err:
            print("Unable to import from cfod and/or chime_frb_api")
            print("Please ensure thoses packagea are installed.")
            print(err)

        # perform an initial get of data from the L4 database in order to
        master = FRBMaster()
        event_L4 =  master.events.get_event(eventid, full_header=True)
        ids, snrs = [], []

        for current_entry_L4 in event_L4["event_beam_header"]:
            ids += [int(current_entry_L4["beam_no"])]
            snrs += [float(current_entry_L4["snr"])]

        # now order id list in descending order based on S/N values.
        snrs = np.array(snrs)
        ids_sorted = [ids[idx] for idx in np.argsort(-snrs).tolist()]
        beam_no = ids_sorted[beam_id]

        # next, perform a GET to retrieve FRBMaster data.
        event = master.events.get_event(eventid)
        entry_realtime = None

        for current_entry in event["measured_parameters"]:
            if current_entry["pipeline"]["name"] == "realtime":
                entry_realtime = current_entry

        print("realtime entry:", entry_realtime)

        # grab l1 data of basic properties, stash into parameter attribute.
        timestamp_substr = entry_realtime["datetime"]

        if "UTC" in timestamp_substr:
            elems = timestamp_substr.split()
            timestamp_substr = " ".join(elems[:len(elems)-1])

        self.burst_parameters["L1"] = {}
        self.burst_parameters["L1"]["dm"] = entry_realtime["dm"]
        self.burst_parameters["L1"]["dm_range"] = entry_realtime["dm_error"]
        self.burst_parameters["L1"]["time_range"] = 0.01
        self.burst_parameters["L1"]["timestamp_fpga"] = event["fpga_time"]
        self.burst_parameters["L1"]["timestamp_utc"] = datetime.datetime.strptime(
            timestamp_substr, "%Y-%m-%d %H:%M:%S.%f"
        )

        # try getting data from frb-vsop.chime.
        print(
            "... trying to grab chime/frb data from fitburst/dm-pipeline results...",
            end="",
        )

        # establish connection to fRBMaster.
        locked_id_dm = None
        locked_id_fitburst = None
        self.burst_parameters["dm-pipeline"] = {}
        self.burst_parameters["fitburst"] = {}

        try:
            if "intensity-dm-pipeline" in event["locked"].keys():
                locked_id_dm = event["locked"]["intensity-dm-pipeline"]


            for current_measurement in event["measured_parameters"]:
                # if there are locked DM-pipeline results, grab and stash those.

                if (
                    current_measurement["pipeline"]["name"] == "intensity-dm-pipeline"
                    and current_measurement["measurement_id"] == locked_id_dm
                ):

                    self.burst_parameters["dm-pipeline"]["snr"] = [
                        current_measurement["snr"]
                    ]
                    self.burst_parameters["dm-pipeline"]["beam_number"] = [
                        current_measurement["beam_number"]
                    ]
                    self.burst_parameters["dm-pipeline"]["dm"] = [
                        current_measurement["dm_snr"]
                    ]
                    self.burst_parameters["dm-pipeline"]["width"] = [
                        current_measurement["width"]
                    ]

                    # get timestamp and avoid error with UTC substring.
                    timestamp_substr = current_measurement["datetime"]

                    if "UTC" in timestamp_substr:
                        elems = timestamp_substr.split()
                        timestamp_substr = " ".join(elems[:len(elems)-1])

                    self.burst_parameters["dm-pipeline"]["timestamp_utc"] = \
                        [datetime.datetime.strptime(
                            str(timestamp_substr), "%Y-%m-%d %H:%M:%S.%f")
                        ]

                    

        except Exception as exc:
            print(
                "WARNING: unable to retrieve locked parameters for DM pipeline from FRBMaster\n",
                "Exception: ",
                exc
            )

        try:
            if "intensity-fitburst" in event["locked"].keys():
                locked_id_fitburst = event["locked"]["intensity-fitburst"]

            for current_measurement in event["measured_parameters"]:

                if (
                    current_measurement["pipeline"]["name"] == "intensity-fitburst"
                    and current_measurement["measurement_id"] == locked_id_fitburst
                ):

                    # determine which fitburst round it is and stash separately.
                    current_round = "round_1"

                    if "Round 1" in current_measurement["pipeline"]["logs"]:
                        pass

                    elif "Round 2" in current_measurement["pipeline"]["logs"]:
                        current_round = "round_2"

                    elif "Round 3" in current_measurement["pipeline"]["logs"]:
                        current_round = "round_3"

                    # now stash fitburst parameters.
                    self.burst_parameters["fitburst"][current_round] = {}
                    self.burst_parameters["fitburst"][current_round][
                        "dm"
                    ] = current_measurement["sub_burst_dm"]
                    self.burst_parameters["fitburst"][current_round][
                        "burst_width"
                    ] = current_measurement["sub_burst_width"]
                    self.burst_parameters["fitburst"][current_round][
                        "amplitude"
                    ] = np.log10(current_measurement["sub_burst_fluence"]).tolist()
                    self.burst_parameters["fitburst"][current_round][
                        "arrival_time"
                    ] = current_measurement["sub_burst_timestamp"]
                    self.burst_parameters["fitburst"][current_round][
                        "spectral_index"
                    ] = current_measurement["sub_burst_spectral_index"]
                    self.burst_parameters["fitburst"][current_round][
                        "spectral_running"
                    ] = current_measurement["sub_burst_spectral_running"]
                    self.burst_parameters["fitburst"][current_round][
                        "ref_freq"
                    ] = [current_measurement["fitburst_reference_frequency"]]

                    # if current round has scattering timescale, stash it as well.
                    if "sub_burst_scattering_timescale" in current_measurement:
                        self.burst_parameters["fitburst"][current_round][
                            "scattering_timescale"
                        ] = current_measurement["sub_burst_scattering_timescale"]
            print("success!")

        except Exception as exc:
            print(
                "WARNING: unable to retrieve locked parameters from frb-vsop.chime:8001\n",
                "Exception: ",
                exc
            )

        # now grab filenames.
        print("... now grabbing locations on the CHIME/FRB archivers...", end="")
        date_string = self.burst_parameters["L1"]["timestamp_utc"].strftime("%Y/%m/%d")
        path_to_data = "{0}/intensity/raw/{1}/astro_{2}/{3:04d}".format(
            mountpoint, date_string, eventid, beam_no
        )

        self.files = glob.glob("{0}/*.msgpack".format(path_to_data))
        self.files.sort(key=natural_keys)
        print("success!")
