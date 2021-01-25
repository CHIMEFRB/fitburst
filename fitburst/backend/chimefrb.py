from chime_frb_api.backends.frb_master import FRBMaster
from cfod.chime_intensity import natural_keys, unpack_datafiles
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

    def __init__(self, eventid, data_location="/data/frb-archiver"):

        # before anything else, ensure eventid makes sense.
        self.eventid = eventid
        assert(isinstance(self.eventid, int))
        print("CHIMEFRBReader executed:")

        # define parameters to be updated by data-retrieval method.
        self.burst_parameters = {}
        self.data_full = None
        self.data_weights = None
        self.data_windowed = None
        self.files = []
        self.frbmaster_request_status = None
        self.freqs = None
        self.downsample_factor = None
        self.fpga_count_start = None
        self.fpga_count_total = None
        self.fpga_frame0_nano = None
        self.rfi_freq_count = None
        self.rfi_mask = None
        self.times = None

        # as a default, grab data from FRBMaster.
        print("... grabbing metadata for eventID {0}".format(self.eventid))
        self._retrieve_metadata_frbmaster(self.eventid)

    def get_parameters(self):
        """
        Returns a dictionary containing parameters as keys and their FRBMaster entries 
        stored as values.
        """

        parameter_dict = {}

        if ("fitburst" in self.burst_parameters):
            for current_key in self.burst_parameters["fitburst"]["round_2"].keys():
                parameter_dict[current_key] = self.burst_parameters["fitburst"]["round_2"][current_key]

        elif ("dm-pipeline" in self.burst_parameters):
            print("woohoo DM pipleine")

        elif ("L1" in self.burst_parameters):
            print("ok at least there is L1")

        else:
            sys.exit("ERROR: no parameters retrieved from FRBMaster!")

        return parameter_dict

    def load_data(self, files, msgpack=True):
        """
        """
        
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
            telescopes["chimefrb"]["num_frames_per_sample"] * \
            telescopes["chimefrb"]["num_factor_upchannel"]
        )
        times += self.fpga_count_start[0]
        self.times = times * telescopes["chimefrb"]["fpga"]["time_per_sample"]

        # now derive frequency information.
        freqs = np.arange(n_freqs, dtype=np.float64)
        freqs *= -(
             telescopes["chimefrb"]["bandwidth"] / \
             telescopes["chimefrb"]["num_channels"]
        )
        freqs += (
            telescopes["chimefrb"]["fpga"]["freq_top"] + \
            telescopes["chimefrb"]["bandwidth"] / \
            telescopes["chimefrb"]["fpga"]["num_channels"] / \
            2
        )
        self.freqs = freqs[::-1]

    def _retrieve_metadata_frbmaster(
            self,
            eventid,
            beam_id=0,
            mountpoint="/data/frb-archiver",
            use_locked=False
        ):
        """
        """

        # perform a GET to retrieve FRBMaster data.
        url_get = "https://frb.chimenet.ca/chimefrb/astro_events/fetch_event_header/{}/".format(
            eventid
        )
        response = requests.request("GET", url_get, auth=("frb", "flub raw burden"))

        # if event is not found in database, then exit.
        self.frbmaster_request_status = response.status_code

        if (self.frbmaster_request_status == 500):
            raise(
                requests.exceptions.ConnectionError("unable to find event {0} in FRBMaster.".format(
                    eventid
                    )
                )
            )

        elif (self.frbmaster_request_status == 400):
            raise(
                requests.exceptions.ConnectionError("unable to connect to FRBMaster.".format(
                    eventid
                   )
                )
            )

        # now start by first grabbing bean and S/N data.
        data = response.json()
        beam_list_dict = data['event_no'][str(eventid)]['event_beam_header']

        snrs = []
        for beam_data in beam_list_dict:
            snrs += [beam_data["snr"]]

        print("... there are {0} beams for this event".format(len(snrs)))

        # sort beam data in descending order of S/N.
        snr_ord_idx = np.argsort(snrs)[::-1]
        beam_data = beam_list_dict[snr_ord_idx[beam_id]]
        beam_no = int(beam_data["beam_no"])

        # grab L1 data of basic properties, stash into parameter attribute.
        self.burst_parameters["L1"] = {}
        self.burst_parameters["L1"]["dm"] = beam_data["dm"]
        self.burst_parameters["L1"]["dm_range"] = beam_data["dm_error"]
        self.burst_parameters["L1"]["time_range"] = beam_data["time_error"]
        self.burst_parameters["L1"]["timestamp_fpga"] = beam_data["timestamp_fpga"]
        self.burst_parameters["L1"]["timestamp_utc"] = datetime.datetime.strptime(
            beam_data["timestamp_utc"], 
            "%Y%m%d%H%M%S.%f"
        )
        
        # try getting data from frb-vsop.chime.
        print("... trying to grab CHIME/FRB data from fitburst/DM-pipeline results...", end="")

        try:
            url_get = "http://frb-vsop.chime:8001"
            master = FRBMaster(base_url=url_get)
            event = master.events.get_event(eventid)
            locked_id_fitburst = event["locked"]["intensity-fitburst"]
            locked_id_dm = event["locked"]["intensity-dm-pipeline"]
            self.burst_parameters["dm-pipeline"] = {}
            self.burst_parameters["fitburst"] = {}

            for current_measurement in event["measured_parameters"]:

                # if there are locked DM-pipeline results, grab and stash those.
                if (current_measurement["pipeline"]["name"] == "intensity-dm-pipeline" and
                    current_measurement["measurement_id"] == locked_id_dm):

                    self.burst_parameters["dm-pipeline"]["snr"] = \
                        [current_measurement["snr"]]
                    self.burst_parameters["dm-pipeline"]["beam_number"] = \
                        [current_measurement["beam_number"]]
                    self.burst_parameters["dm-pipeline"]["dm"] = \
                        [current_measurement["dm_snr"]]
                    self.burst_parameters["dm-pipeline"]["width"] = \
                        [current_measurement["width"]]
                    #self.burst_parameters["dm-pipeline"]["timestamp_utc"] = \
                    #    [datetime.datetime.strptime(
                    #        str(current_measurement["datetime"]),
                    #        "%Y-%m-%d %H:%M:%S.%f %Z%z",
                    #    )
                    #]
                    
                elif (current_measurement["pipeline"]["name"] == "intensity-fitburst" and 
                    current_measurement["measurement_id"] == locked_id_fitburst):

                    # determine which fitburst round it is and stash separately.
                    current_round = "round_1"

                    if ("Round 1" in current_measurement["pipeline"]["logs"]):
                        pass

                    elif (("Round 2" in current_measurement["pipeline"]["logs"])):
                        current_round = "round_2"

                    elif (("Round 3" in current_measurement["pipeline"]["logs"])):
                        current_round = "round_3"

                    # now stash fitburst parameters.
                    self.burst_parameters["fitburst"][current_round] = {}
                    self.burst_parameters["fitburst"][current_round]["dm"] = \
                        current_measurement["sub_burst_dm"]
                    self.burst_parameters["fitburst"][current_round]["width"] = \
                        current_measurement["sub_burst_width"]
                    self.burst_parameters["fitburst"][current_round]["amplitude"] = \
                        current_measurement["sub_burst_fluence"]
                    self.burst_parameters["fitburst"][current_round]["arrival_time"] = \
                        current_measurement["sub_burst_timestamp"]
                    self.burst_parameters["fitburst"][current_round]["spectral_index"] = \
                        current_measurement["sub_burst_spectral_index"]
                    self.burst_parameters["fitburst"][current_round]["spectral_running"] = \
                        current_measurement["sub_burst_spectral_running"]
                    self.burst_parameters["fitburst"][current_round]["reference_freq"] = \
                        current_measurement["fitburst_reference_frequency"]

                    # if current round has scattering timescale, stash it as well.
                    if ("scattering_timescale" in current_measurement):
                        self.burst_parameters["fitburst"][current_round]["timestamp_scattering"] = [
                            current_measurement["sub_burst_scattering_timescale"]
                        ]
            print("success!")

        except Exception as exc:
            print("WARNING: unable to retrieve locked parameters from frb-vsop.chime:8001")

        # now grab filenames.
        print("... now grabbing locations on the CHIME/FRB archivers...", end="")
        date_string = self.burst_parameters["L1"]["timestamp_utc"].strftime("%Y/%m/%d")
        path_to_data = "{0}/{1}/astro_{2}/intensity/raw/{3:04d}".format(
            mountpoint, 
            date_string,
            eventid,
            beam_no
        )

        self.files = glob.glob("{0}/*.msgpack".format(path_to_data))
        self.files.sort(key=natural_keys)
        print("success!")
