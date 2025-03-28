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

    def __init__(self, ephys_roi_id):
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
        self.load_metadata()

    def load_metadata(self):
        """Load metadata from jsons"""
        self.json_dicts = read_json_files(self.ephys_roi_id)
        self.df_sweeps = jsons_to_df(self.json_dicts)

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


if __name__ == "__main__":

    logging.basicConfig(level=logging.INFO)

    # Test the class
    ephys_roi_id = "1410790193"
    raw = PatchSeqNWB(ephys_roi_id)

    print(raw.get_raw_trace(0))  # Get the raw trace for the first sweep
    print(raw.get_stimulus(0))  # Get the stimulus for the first sweep
