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
    ].values[:3]

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


def save_to_hdf5(output_path, df_features, df_sweeps, traces, valid_sweep_numbers, ephys_roi_id):
    """Save data to HDF5 format.
    
    Args:
        output_path (str): Path to save the HDF5 file
        df_features (pd.DataFrame): DataFrame containing eFEL features
        df_sweeps (pd.DataFrame): DataFrame containing sweep metadata
        traces (list): List of trace dictionaries containing time and voltage data
        valid_sweep_numbers (np.ndarray): Array of valid sweep numbers
        ephys_roi_id (str): The ephys ROI ID
    """
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    with h5py.File(output_path, 'w') as f:
        # Save metadata
        f.attrs['ephys_roi_id'] = ephys_roi_id
        
        # Save features DataFrame
        features_group = f.create_group('features')
        for col in df_features.columns:
            features_group.create_dataset(col, data=df_features[col].values)
        features_group.attrs['index'] = list(df_features.index)
        
        # Save sweep metadata
        sweeps_group = f.create_group('sweeps')
        for col in df_sweeps.columns:
            if df_sweeps[col].dtype == 'object':
                # Convert object columns to strings for HDF5 storage
                sweeps_group.create_dataset(col, data=df_sweeps[col].astype(str).values)
            else:
                sweeps_group.create_dataset(col, data=df_sweeps[col].values)
        
        # Save traces
        traces_group = f.create_group('traces')
        for i, trace in enumerate(traces):
            trace_group = traces_group.create_group(f'sweep_{valid_sweep_numbers[i]}')
            trace_group.create_dataset('time', data=trace['T'])
            trace_group.create_dataset('voltage', data=trace['V'])
            trace_group.create_dataset('stim_start', data=trace['stim_start'])
            trace_group.create_dataset('stim_end', data=trace['stim_end'])
        
        # Save valid sweep numbers
        f.create_dataset('valid_sweep_numbers', data=valid_sweep_numbers)
    
    logger.info(f"Saved data to {output_path}")

def reformat_features(features):
    """Reformat features extracted from eFEL.
    
    This function processes the raw features dictionary and creates two DataFrames:
    1. A per-sweep DataFrame with scalar values (single values per sweep)
    2. A per-spike DataFrame for features that have multiple values per sweep
    
    For multi-spike features, the first spike's value is also stored in the per-sweep
    DataFrame with a 'first_spike_' prefix.
    
    Args:
        features (dict): Dictionary of features extracted by eFEL
        
    Returns:
        dict: Dictionary containing reformatted DataFrames and interpolated data
    """
    df_features = pd.DataFrame(features, index=valid_sweep_numbers)
    df_features.index.name = "sweep_number"

    # Create a new DataFrame for per-spike features
    list_features_per_spike = []

    # Create a new DataFrame for per-sweep features (with scalar values)
    dict_features_per_sweep = {}

    # Pop two columns "time" and "voltage" from df_features and save to "interpolated_time" and "interpolated_voltage"
    interpolated_time = df_features["time"]
    interpolated_voltage = df_features["voltage"]
    df_features.drop(columns=["time", "voltage"], inplace=True)

    # Process each column in the original DataFrame
    for col in df_features.columns:
        lengths = df_features[col].map(lambda x: 0 if x is None else len(x))
        
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
    
    return {
        "df_features_per_sweep": df_features_per_sweep,
        "df_features_per_spike": df_features_per_spike,
        "interpolated_time": interpolated_time,
        "interpolated_voltage": interpolated_voltage
    }


if __name__ == "__main__":
    import LCNE_patchseq_analysis.efel  # Apply global efel settings
    
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
    
    # Reformat features
    features_dict = reformat_features(features)
    
    # Save features to HDF5

