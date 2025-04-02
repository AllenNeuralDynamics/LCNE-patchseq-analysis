"""Extracting features from a single cell."""

import logging

import efel
import pandas as pd
import numpy as np

from LCNE_patchseq_analysis import TIME_STEP
from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB
from LCNE_patchseq_analysis.efel.io import save_dict_to_hdf5, load_dict_from_hdf5

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
            "sweep_number": [sweep_number],
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

    # Pop time and voltage columns and save to interpolated data if requested
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
            dict_features_per_sweep[col] = df_features[col].apply(
                lambda x: x[0] if x is not None and len(x) > 0 else None
            )
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
    df_features_per_spike = pd.DataFrame(list_features_per_spike).pivot(
        index=["sweep_number", "spike_idx"], 
        columns="feature", 
        values="value"
    )
    
    dict_to_save = {
        "df_features_per_sweep": df_features_per_sweep,
        "df_features_per_spike": df_features_per_spike,
    }
    
    if if_save_interpolated:
        dict_to_save["interpolated_time"] = interpolated_time
        dict_to_save["interpolated_voltage"] = interpolated_voltage
    
    return dict_to_save


def extract_spike_waveforms(traces, features_dict, spike_window: tuple = (-5, 10)):
    """Extract spike waveforms from raw data.
    
    Args:
        traces: raw traces
        features_dict: Dictionary containing features extracted by eFEL
        spike_window: Tuple of two integers, the start and end of the spike window
            in milliseconds relative to the peak time
    """
    peak_times = features_dict["df_features_per_spike"].reset_index(
        ).set_index("sweep_number")["peak_time"]
    
    # Time can be determined by the sampling rate
    t_aligned = np.arange(spike_window[0], spike_window[1], step=TIME_STEP)
    vs = []
    
    for trace in traces:
        if trace["sweep_number"][0] not in peak_times.index:
            continue    
        peak_times_this_sweep = peak_times.loc[trace["sweep_number"]]
        t = trace["T"]
        v = trace["V"]
        
        for peak_time in peak_times_this_sweep:
            idx = np.where((t >= peak_time + spike_window[0]) & (t < peak_time + spike_window[1]))[0]
            v_this = v[idx]
            vs.append(v_this)
            
    df_spike_waveforms = pd.DataFrame(vs, 
                                      index=features_dict["df_features_per_spike"].index,
                                      columns=pd.Index(t_aligned, name="ms_to_peak"),
                                      )
    return df_spike_waveforms


def extract_features_using_efel(raw, if_save_interpolated):
    """Extract features using eFEL."""

    # -- Package all valid sweeps for eFEL --
    traces, valid_sweep_numbers = pack_traces_for_efel(raw)
      
    # Get all features
    logger.debug(f"Getting features for {len(traces)} traces...")
    features = efel.get_feature_values(
        traces=traces,
        feature_names=efel.get_feature_names(),  # Get all features
        raise_warnings=False,
    )
    logger.debug("Done!")
    
    # Reformat features
    df_features = pd.DataFrame(features, index=valid_sweep_numbers)
    df_features.index.name = "sweep_number"
    features_dict = reformat_features(df_features, if_save_interpolated)
    
    # -- Extract spike waveforms --
    df_spike_waveforms = extract_spike_waveforms(traces, features_dict)

    # -- Enrich df_sweeps --
    df_sweeps = raw.df_sweeps.copy()
    df_sweeps.insert(0, "ephys_roi_id", ephys_roi_id)
    col_to_df_sweeps = {
        "spike_count": "efel_num_spikes",
        "first_spike_AP_width": "efel_first_spike_AP_width",
    }
    _df_to_df_sweeps = (
        features_dict["df_features_per_sweep"][list(col_to_df_sweeps.keys())]
        .rename(columns=col_to_df_sweeps)
    )
    df_sweeps = df_sweeps.merge(_df_to_df_sweeps, on="sweep_number", how="left")
                   
    # Add metadata to features_dict
    features_dict["df_sweeps"] = df_sweeps
    features_dict["df_spike_waveforms"] = df_spike_waveforms
    features_dict["efel_settings"] = pd.Series([efel.get_settings().__dict__])
    
    return features_dict


def process_one_nwb(ephys_roi_id: str, if_save_interpolated: bool = False):
    # --- 1. Get raw data ---
    raw = PatchSeqNWB(ephys_roi_id=ephys_roi_id)

    # --- 2. Extract features using eFEL ---
    features_dict = extract_features_using_efel(raw, if_save_interpolated)

    # --- 3. Save features_dict to HDF5 using panda's hdf5 store ---
    save_dict_to_hdf5(features_dict, f"data/efel_features/{ephys_roi_id}_efel_features.h5")

    return


if __name__ == "__main__":
    import tqdm
    
    logging.basicConfig(level=logging.INFO)

    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata

    df_meta = load_ephys_metadata()
    
    for ephys_roi_id in tqdm.tqdm(df_meta["ephys_roi_id_tab_master"][:1]):
        logger.info(f"Processing {ephys_roi_id}...")
        process_one_nwb(ephys_roi_id=str(int(ephys_roi_id)), 
                        if_save_interpolated=False)
 
