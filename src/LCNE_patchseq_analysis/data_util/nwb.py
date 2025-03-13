"""Get raw data traces from NWB files."""

import glob
import h5py

from LCNE_patchseq_analysis import RAW_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import read_json_files, jsons_to_df


class PatchSeqNWB():
    """Class for accessing patch-seq NWB files using h5py."""
    
    SAMPLING_RATE = 50000  # Hard-coded sampling rate for patch-seq data
    dt_ms = 1 / SAMPLING_RATE * 1000
    
    def __init__(self, ephys_roi_id):
        self.ephys_roi_id = ephys_roi_id
        self.raw_path_this = f"{RAW_DIRECTORY}/Ephys_Roi_Result_{ephys_roi_id}"
        self.nwbs = glob.glob(f"{self.raw_path_this}/*spikes.nwb")
        if len(self.nwbs) == 0:
            raise FileNotFoundError(f"No NWB files found for {ephys_roi_id}")
        
        # Load nwb
        self.hdf = h5py.File(self.nwbs[0], 'r')
        
        # Load metadata from jsons
        self.json_dicts = read_json_files(ephys_roi_id)
        self.df_sweep = jsons_to_df(self.json_dicts)
    
    def get_raw_trace(self, sweep_number):
        """Get the raw trace for a given sweep number."""
        try:
            return self.hdf[f"acquisition/data_{sweep_number:05}_AD0/data"]
        except KeyError:
            raise KeyError(f"Sweep number {sweep_number} not found in NWB file.")

    def get_stimulus(self, sweep_number):
        """Get the stimulus trace for a given sweep number."""
        try:
            return self.hdf[f"stimulus/presentation/data_{sweep_number:05}_DA0/data"]
        except KeyError:
            raise KeyError(f"Sweep number {sweep_number} not found in NWB file.")



if __name__ == "__main__":
    # Test the class
    ephys_roi_id = "1410790193"
    raw = PatchSeqNWB(ephys_roi_id)
    print(f"Found {len(raw.nwbs)} NWB files for {ephys_roi_id}")
    print(raw.hdf.keys())
    print(raw.hdf["stimulus/presentation/data_00000_DA0/data"])
    
    # Test the data extraction
    print(raw.hdf["acquisition/data_00000_AD0/data"])
    print(raw.hdf["stimulus/presentation/data_00000_DA0/data"])
    
    print(raw.df_sweep.head(10))  # Check the first 10 rows of the merged dataframe