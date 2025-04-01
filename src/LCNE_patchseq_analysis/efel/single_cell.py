"""Extracting features from a single cell."""

import logging
import multiprocessing as mp

import efel
import numpy as np
import pandas as pd

from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB

logger = logging.getLogger(__name__)
    

def pack_traces_for_efel(raw):
    """Package traces for eFEL."""
    # Only valid sweeps (passed not NA, and exclude CHIRP sweeps)
    df_sweeps = raw.df_sweeps
    valid_sweep_numbers = df_sweeps.loc[
        (df_sweeps["passed"].notna()) &
        ~(df_sweeps["stimulus_code"].str.contains("CHIRP", case=False)),
        "sweep_number"
    ].values

    traces = []
    for sweep_number in valid_sweep_numbers:
        trace = raw.get_raw_trace(sweep_number)
        time = raw.get_time(sweep_number)

        meta_this = df_sweeps.query("sweep_number == @sweep_number")
        stim_start = meta_this["stimulus_start_time"].values[0]
        stim_end = stim_start + meta_this["stimulus_duration"].values[0]

        # Use efel to get features
        traces.append({
            "T": time,
            "V": trace,
            "stim_start": [stim_start],
            "stim_end": [stim_end],
        })

    logger.info(f"Packed {len(traces)} traces for eFEL.")
    return traces, valid_sweep_numbers

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    ephys_roi_id = "1212557784"
    
    # Get raw data
    raw = PatchSeqNWB(ephys_roi_id=ephys_roi_id)
    
    # Package all valid sweeps for eFEL
    traces, valid_sweep_numbers = pack_traces_for_efel(raw)
    
    # Get all features
    logger.info(f"Getting features for {len(traces)} traces...")
    features = efel.get_feature_values(
        traces,
        efel.get_feature_names(),  # Get all features
        raise_warnings=False,
    )
    logger.info("Done!")
    # Save features
    df_features = pd.DataFrame(features, index=valid_sweep_numbers)
    

    
    
