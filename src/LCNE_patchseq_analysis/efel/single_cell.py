"""Extracting features from a single cell."""

import efel
import numpy as np
import pandas as pd

from LCNE_patchseq_analysis.efel import EFEL_SETTINGS
from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB


def efel_single_sweep(ephys_roi_id, sweep_number):

    
    # Get trace and time for sweep 0 
    trace = raw.get_raw_trace(sweep_number)
    time = raw.dt_ms * np.arange(len(trace))
    stimulus = raw.get_stimulus(sweep_number)
    
    meta_this = raw.df_sweeps.query("sweep_number == @sweep_number").copy()
    meta_this.loc[:, "ephys_roi_id"] = ephys_roi_id
    stim_code = meta_this["stimulus_code"].values[0]
    stim_start = meta_this["stimulus_start_time"].values[0] * 1000
    stim_end = stim_start + meta_this["stimulus_duration"].values[0] * 1000

    # Use efel to get features
    traces = [
        {
            "T": time,
            "V": trace,
            "stim_start": [stim_start],
            "stim_end": [stim_end],
        }
    ]
    df_features = pd.DataFrame(efel.get_feature_values(
        traces, efel.get_feature_names(), raise_warnings=False))
    
    return {
        "stim_code": stim_code,
        "time": time,
        "trace": trace,
        "meta": meta_this,
        "stimulus": stimulus,
        "stim_start": stim_start,
        "stim_end": stim_end}, df_features
    

if __name__ == "__main__":
    ephys_roi_id = "1212557784"
    sweep_number = 9
    
    raw = PatchSeqNWB(ephys_roi_id=ephys_roi_id)
    df_sweeps = raw.df_sweeps
    
    # Look for valid sweeps
    valid_sweeps = df_sweeps.loc[df_sweeps["passed"].notna(), "sweep_number"]
    
    raw, df_features = efel_single_sweep(ephys_roi_id, sweep_number)
    print(df_features)