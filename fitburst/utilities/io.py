from chime_frb_api.backends.frb_master import FRBMaster
from . import bases
import numpy as np
import datetime
import requests
import glob
import pytz
import sys

class CHIMEFRBReader(bases.ReaderBaseClass):
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
        self.files = []
        self.frbmaster_request_status = None

        # as a default, grab data from FRBMaster.
        print("... grabbing metadata for eventID {0}".format(self.eventid))
        self._retrieve_metadata_frbmaster(self.eventid)

    def load_data(self, files, msgpack=True):
        """
        """
        
        pass

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
        url_get = "https://frb.chimenet.ca/chimefrb/astro_events/fetch_event_header/{}/".format(eventid)
        response = requests.request("GET", url_get, auth=("frb", "flub raw burden"))

        # if event is not found in database, then exit.
        self.frbmaster_request_status = response.status_code

        if (self.frbmaster_request_status == 500):
            raise(requests.exceptions.ConnectionError("unable to find event {0} in FRBMaster.".format(eventid)))

        elif (self.frbmaster_request_status == 400):
            raise(requests.exceptions.ConnectionError("unable to connect to FRBMaster.".format(eventid)))

        # now start by first grabbing bean and S/N data.
        data = response.json()
        beam_list_dict = data['event_no'][str(eventid)]['event_beam_header']

        snrs = []
        for beam_data in beam_list_dict:
            snrs += [beam_data["snr"]]

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
        try:
            print("hey!")
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
                    
                    print("yes!")

                elif (current_measurement["pipeline"]["name"] == "intensity-fitburst" and 
                    current_measurement["measurement_id"] == locked_id_fitburst):

                    print("yo")
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
                    self.burst_parameters["fitburst"][current_round]["timestamp_utc"] = \
                        current_measurement["sub_burst_timestamp"]
                    self.burst_parameters["fitburst"][current_round]["spectral_index"] = \
                        current_measurement["sub_burst_spectral_index"]
                    self.burst_parameters["fitburst"][current_round]["spectral_running"] = \
                        current_measurement["sub_burst_spectral_running"]

                    # if current round has scattering timescale, stash it as well.
                    if ("scattering_timescale" in current_measurement):
                        self.burst_parameters["fitburst"][current_round]["timestamp_scattering"] = [
                            current_measurement["sub_burst_scattering_timescale"]
                        ]

        except Exception as exc:
            print("WARNING: unable to retrieve locked parameters from frb-vsop.chime:8001")

        # now grab filenames.
        date_string = self.burst_parameters["L1"]["timestamp_utc"].strftime("%Y/%m/%d")
        path_to_data = "{0}/{1}/astro_{2}/intensity/raw/{3:04d}".format(
            mountpoint, 
            date_string,
            eventid,
            beam_no
        )

        self.files = glob.glob("{0}/*.msgpack".format(path_to_data))
