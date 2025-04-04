"""Get raw data traces from NWB files."""

import glob
import logging

import h5py
import numpy as np

from LCNE_patchseq_analysis import RAW_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import jsons_to_df, read_json_files

logger = logging.getLogger(__name__)


class PatchSeqNWB:
    """Class for accessing patch-seq NWB files using h5py."""

    SAMPLING_RATE = 50000  # Hard-coded sampling rate for patch-seq data
    dt_ms = 1 / SAMPLING_RATE * 1000

    def __init__(self, ephys_roi_id, if_load_metadata=True):
        """Initialization using the ephys_roi_id"""
        self.ephys_roi_id = ephys_roi_id
        self.raw_path_this = f"{RAW_DIRECTORY}/Ephys_Roi_Result_{ephys_roi_id}"
        nwbs = glob.glob(f"{self.raw_path_this}/*.nwb")
        self.nwbs = [f for f in nwbs if "spike" not in f]

        if len(self.nwbs) == 0:
            raise FileNotFoundError(f"No NWB files found for {ephys_roi_id}")

        if len(self.nwbs) > 1:
            raise ValueError(f"Multiple NWB files found for {ephys_roi_id}")

        # Load nwb
        logger.info(f"Loading NWB file {self.nwbs[0]}")
        self.hdf = h5py.File(self.nwbs[0], "r")
        self.n_sweeps = len(self.hdf["acquisition"])

        # Load metadata
        if if_load_metadata:
            self.load_metadata()

    def load_metadata(self):
        """Load metadata from jsons"""
        self.json_dicts = read_json_files(self.ephys_roi_id)
        self.df_sweeps = jsons_to_df(self.json_dicts)

        if self.df_sweeps is None:
            logger.warning(f"No sweep features found for {self.ephys_roi_id}!")
            self.valid_sweeps = []
            return

        # Turn start_time and duration into ms
        self.df_sweeps["stimulus_start_time"] = self.df_sweeps["stimulus_start_time"] * 1000
        self.df_sweeps["stimulus_duration"] = self.df_sweeps["stimulus_duration"] * 1000
        self.valid_sweeps = self.df_sweeps.loc[self.df_sweeps["passed"].notna(), "sweep_number"]

    def get_raw_trace(self, sweep_number):
        """Get the raw trace for a given sweep number."""
        try:
            return np.array(self.hdf[f"acquisition/data_{sweep_number:05}_AD0/data"])
        except KeyError:
            raise KeyError(f"Sweep number {sweep_number} not found in NWB file.")

    def get_stimulus(self, sweep_number):
        """Get the stimulus trace for a given sweep number."""
        try:
            return np.array(self.hdf[f"stimulus/presentation/data_{sweep_number:05}_DA0/data"])
        except KeyError:
            raise KeyError(f"Sweep number {sweep_number} not found in NWB file.")

    def get_time(self, sweep_number):
        """Get the time for a given sweep number."""
        try:
            length = len(self.hdf[f"acquisition/data_{sweep_number:05}_AD0/data"])
            return self.dt_ms * np.arange(length)
        except KeyError:
            raise KeyError(f"Sweep number {sweep_number} not found in NWB file.")


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Test the class
    ephys_roi_id = "1410790193"
    raw = PatchSeqNWB(ephys_roi_id)

    print(len(raw.get_raw_trace(0)))  # Get the raw trace for the first sweep
    print(len(raw.get_stimulus(0)))  # Get the stimulus for the first sweep
    print(len(raw.get_time(0)))  # Get the time for the first sweep
