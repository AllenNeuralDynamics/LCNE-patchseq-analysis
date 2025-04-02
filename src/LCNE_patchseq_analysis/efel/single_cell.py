"""Extracting features from a single cell."""

import logging
import multiprocessing as mp
import os

import efel
import numpy as np
import pandas as pd
import h5py

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


def reformat_features(df_features, if_save_interpolated: bool = False):
    """Reformat features extracted from eFEL.
    
    This function processes the raw features dictionary and creates two DataFrames:
    1. A per-sweep DataFrame with scalar values (single values per sweep)
    2. A per-spike DataFrame for features that have multiple values per sweep
    
    For multi-spike features, the first spike's value is also stored in the per-sweep
    DataFrame with a 'first_spike_' prefix.
    
    Args:
        features (dict): Dictionary of features extracted by eFEL
        if_save_interpolated (bool): Whether to save the interpolated data
            By default, interp_step is set to 0.02 ms, which is the same as the sampling rate.
            So there is no need to save the interpolated data.
        
    Returns:
        dict: Dictionary containing reformatted DataFrames and interpolated data
    """


    # Create a new DataFrame for per-spike features
    list_features_per_spike = []

    # Create a new DataFrame for per-sweep features (with scalar values)
    dict_features_per_sweep = {}

    # Pop two columns "time" and "voltage" from df_features and save to "interpolated_time" and "interpolated_voltage"
    if if_save_interpolated:
        interpolated_time = df_features["time"]
        interpolated_voltage = df_features["voltage"]
    df_features.drop(columns=["time", "voltage"], inplace=True)

    # Process each column in the original DataFrame
    for col in df_features.columns:
        lengths = df_features[col].map(lambda x: 0 if x is None else len(x))
        
        # If all values are None, skip this column
        if max(lengths) == 0:
            continue
        
        # Check if it's a scalar or array feature
        if max(lengths) == 1:
            # For single values, extract the scalar
            dict_features_per_sweep[col] = df_features[col].apply(lambda x: x[0] if x is not None and len(x) > 0 else None)
        else:
            # For multi-spike features
            # 1. Extract first spike value to per_sweep DataFrame
            dict_features_per_sweep[f"first_spike_{col}"] = df_features[col].apply(
                lambda x: x[0] if x is not None and len(x) > 0 else None
            )
            
            # 2. Expand to per-spike DataFrame
            for sweep_idx, sweep_values in df_features[col].items():
                if sweep_values is not None and len(sweep_values) > 0:
                    list_features_per_spike.extend([
                        {"sweep_number": sweep_idx, "spike_idx": i, "feature": col, "value": val}
                        for i, val in enumerate(sweep_values)
                    ])

    # Pack dataframes
    df_features_per_sweep = pd.DataFrame(dict_features_per_sweep)
    df_features_per_spike = pd.DataFrame(list_features_per_spike).pivot(index=["sweep_number", "spike_idx"], columns="feature", values="value")
    
    dict_to_save = {
        "df_features_per_sweep": df_features_per_sweep,
        "df_features_per_spike": df_features_per_spike,
    }
    
    if if_save_interpolated:
        dict_to_save["interpolated_time"] = interpolated_time
        dict_to_save["interpolated_voltage"] = interpolated_voltage
    
    return dict_to_save
    

def save_dict_to_hdf5(data_dict: dict, filename: str, compress: bool = False):
    """
    Save a dictionary of DataFrames to an HDF5 file using pandas.HDFStore.

    Args:
        data_dict: dict of {str: pd.DataFrame}
        filename: path to .h5 file
        compress: whether to use compression (blosc, level 9)
    """
    with pd.HDFStore(filename, mode='w') as store:
        for key, df in data_dict.items():
            if compress:
                store.put(key, df, format='table', complib='blosc', complevel=9)
            else:
                store.put(key, df)

                
def load_dict_from_hdf5(filename: str):
    """
    Load a dictionary of DataFrames from an HDF5 file using pandas.HDFStore.

    Args:
        filename: path to .h5 file
        
    Returns:
        dict: Dictionary of DataFrames
    """
    with pd.HDFStore(filename, mode='r') as store:
        return {key: store[key] for key in store.keys()}


def process_one_nwb(ephys_roi_id: str, if_save_interpolated: bool = False):
    # Get raw data
    raw = PatchSeqNWB(ephys_roi_id=ephys_roi_id)
   
    # Package all valid sweeps for eFEL
    traces, valid_sweep_numbers = pack_traces_for_efel(raw)
      
    # Get all features
    logger.debug(f"Getting features for {len(traces)} traces...")
    features = efel.get_feature_values(
        traces,
        efel.get_feature_names(),  # Get all features
        raise_warnings=False,
    )
    logger.debug("Done!")
    
    # Reformat features
    df_features = pd.DataFrame(features, index=valid_sweep_numbers)
    df_features.index.name = "sweep_number"
    features_dict = reformat_features(df_features, if_save_interpolated)
    
    # Save features_dict to HDF5 using panda's hdf5 store
    save_dict_to_hdf5(features_dict, f"data/efel_features/{ephys_roi_id}_efel_features.h5")

    # Load features_dict from HDF5
    features_dict_loaded = load_dict_from_hdf5(f"data/efel_features/{ephys_roi_id}_efel_features.h5")
    

if __name__ == "__main__":
    import LCNE_patchseq_analysis.efel  # Apply global efel settings
    import tqdm
    
    logging.basicConfig(level=logging.INFO)

    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata

    df_meta = load_ephys_metadata()
    
    for ephys_roi_id in tqdm.tqdm(df_meta["ephys_roi_id_tab_master"]):
        logger.info(f"Processing {ephys_roi_id}...")
        process_one_nwb(ephys_roi_id=str(int(ephys_roi_id)), 
                        if_save_interpolated=False)
 

    