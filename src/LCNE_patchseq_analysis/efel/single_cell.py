"""Extracting features from a single cell."""

import logging
import os
from typing import Dict, List, Tuple, Optional, Any

import efel
import pandas as pd
import numpy as np

from LCNE_patchseq_analysis import TIME_STEP, RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB
from LCNE_patchseq_analysis.efel.io import save_dict_to_hdf5
from LCNE_patchseq_analysis.efel.plot import plot_sweep_summary

logger = logging.getLogger(__name__)


def pack_traces_for_efel(raw: PatchSeqNWB) -> Tuple[List[Dict[str, Any]], np.ndarray]:
    """Package traces for eFEL.
    
    Args:
        raw: PatchSeqNWB object containing the raw data
        
    Returns:
        Tuple containing:
        - List of trace dictionaries for eFEL
        - Array of valid sweep numbers
    """
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

        traces.append({
            "T": time,
            "V": trace,
            "stim_start": [stim_start],
            "stim_end": [stim_end],
            "sweep_number": [sweep_number],  # not for eFEL, but for future use
        })

    logger.info(f"Packed {len(traces)} traces for eFEL.")
    return traces, valid_sweep_numbers


def reformat_features(
    df_features: pd.DataFrame,
    if_save_interpolated: bool = False
) -> Dict[str, Any]:
    """Reformat features extracted from eFEL.
    
    Args:
        df_features: DataFrame of features extracted by eFEL
        if_save_interpolated: Whether to save the interpolated data
        
    Returns:
        Dictionary containing reformatted DataFrames and interpolated data
    """
    # Create a new DataFrame for per-spike features
    list_features_per_spike = []

    # Create a new DataFrame for per-sweep features (with scalar values)
    dict_features_per_sweep = {}

    # Pop time and voltage columns and save to interpolated data if requested
    interpolated_data = {}
    if if_save_interpolated:
        interpolated_data["interpolated_time"] = df_features["time"]
        interpolated_data["interpolated_voltage"] = df_features["voltage"]
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
                        {
                            "sweep_number": sweep_idx,
                            "spike_idx": i,
                            "feature": col,
                            "value": val
                        }
                        for i, val in enumerate(sweep_values)
                    ])

    # Pack dataframes
    df_features_per_sweep = pd.DataFrame(dict_features_per_sweep)
    df_features_per_spike = pd.DataFrame(list_features_per_spike).pivot(
        index=["sweep_number", "spike_idx"],
        columns="feature",
        values="value"
    )

    result_dict = {
        "df_features_per_sweep": df_features_per_sweep,
        "df_features_per_spike": df_features_per_spike,
    }

    if if_save_interpolated:
        result_dict.update(interpolated_data)

    return result_dict


def extract_spike_waveforms(
    raw_traces: List[Dict[str, Any]],
    features_dict: Dict[str, Any],
    spike_window: Tuple[float, float] = (-5, 10)
) -> pd.DataFrame:
    """Extract spike waveforms from raw data.
    
    Args:
        raw_traces: List of raw trace dictionaries
        features_dict: Dictionary containing features extracted by eFEL
        spike_window: Tuple of two floats, the start and end of the spike window
            in milliseconds relative to the peak time
            
    Returns:
        DataFrame containing spike waveforms
    """
    peak_times = features_dict["df_features_per_spike"].reset_index(
    ).set_index("sweep_number")["peak_time"]

    # Time can be determined by the sampling rate
    t_aligned = np.arange(spike_window[0], spike_window[1], step=TIME_STEP)
    vs = []

    for raw_trace in raw_traces:
        if raw_trace["sweep_number"][0] not in peak_times.index:
            continue
        peak_times_this_sweep = peak_times.loc[raw_trace["sweep_number"]]
        t = raw_trace["T"]
        v = raw_trace["V"]

        for peak_time in peak_times_this_sweep:
            idx = np.where(
                (t >= peak_time + spike_window[0]) &
                (t < peak_time + spike_window[1])
            )[0]
            v_this = v[idx]
            vs.append(v_this)

    return pd.DataFrame(
        vs,
        index=features_dict["df_features_per_spike"].index,
        columns=pd.Index(t_aligned, name="ms_to_peak"),
    )
    

def extract_peri_stimulus_raw_traces(
    raw_traces: List[Dict[str, Any]],
    features_dict: Dict[str, Any],
    before_ratio: float = 0.2,
    after_ratio: float = 0.5,
    min_before_ms: float = 10,
    min_after_ms: float = 100,
) -> pd.DataFrame:
    """Extract peri-stimulus raw traces from raw data.
    """
    
    vs = []
    Is = []
    begin_ts = []
    end_ts = []
    
    for raw_trace in raw_traces:
        v = raw_trace["V"]
        I = raw_trace["stimulus"]
        t = raw_trace["T"]
        stim_start = raw_trace["stim_start"][0]
        stim_end = raw_trace["stim_end"][0]
        
        before_ms = max(min_before_ms, before_ratio * (stim_end - stim_start))
        after_ms = max(min_after_ms, after_ratio * (stim_end - stim_start))
        
        begin_t = max(0, stim_start - before_ms)
        end_t = min(t[-1], stim_end + after_ms)
        idx_before = np.where(t >= begin_t)[0][0]
        idx_after = np.where(t >= end_t)[0][0]
        
        vs.append(v[idx_before:idx_after])
        Is.append(I[idx_before:idx_after])
        begin_ts.append(begin_t)
        end_ts.append(end_t)
        
    df_peri_stimulus_raw_traces = pd.DataFrame(
        {
            "V": vs,
            "I": Is,
            "begin_t": begin_ts,
            "end_t": end_ts,
            "stim_start": [raw_trace["stim_start"][0] for raw_trace in raw_traces],
            "stim_end": [raw_trace["stim_end"][0] for raw_trace in raw_traces],
        },
        index=features_dict["df_features_per_sweep"].index,
    )

    return df_peri_stimulus_raw_traces


def extract_features_using_efel(
    raw: PatchSeqNWB,
    if_save_interpolated: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """Extract features using eFEL.
    
    Args:
        raw: PatchSeqNWB object containing the raw data
        if_save_interpolated: Whether to save interpolated data
        
    Returns:
        Tuple containing:
        - Dictionary of extracted features
        - List of raw traces
    """
    # -- Package all valid sweeps for eFEL --
    raw_traces, valid_sweep_numbers = pack_traces_for_efel(raw)

    # Get all features
    logger.debug(f"Getting features for {len(raw_traces)} traces...")
    features = efel.get_feature_values(
        traces=raw_traces,
        feature_names=efel.get_feature_names(),  # Get all features
        raise_warnings=False,
    )
    logger.debug("Done!")

    # Reformat features
    df_features = pd.DataFrame(features, index=valid_sweep_numbers)
    df_features.index.name = "sweep_number"
    features_dict = reformat_features(df_features, if_save_interpolated)

    # -- Extract spike waveforms --
    df_spike_waveforms = extract_spike_waveforms(raw_traces, features_dict)
    
    # -- Extract peri-stimulus raw traces --
    # Append stimulus to raw_traces (doing here because eFEL cannot handle it)
    for raw_trace in raw_traces:
        raw_trace["stimulus"] = raw.get_stimulus(raw_trace["sweep_number"][0])
    df_peri_stimulus_raw_traces = extract_peri_stimulus_raw_traces(raw_traces, features_dict)

    # -- Enrich df_sweeps --
    df_sweeps = raw.df_sweeps.copy()
    df_sweeps.insert(0, "ephys_roi_id", raw.ephys_roi_id)
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
    features_dict["df_peri_stimulus_raw_traces"] = df_peri_stimulus_raw_traces
    features_dict["efel_settings"] = pd.DataFrame([efel.get_settings().__dict__])

    return features_dict, raw_traces


def process_one_nwb(
    ephys_roi_id: str,
    if_save_interpolated: bool = False,
    save_dir: str = RESULTS_DIRECTORY
) -> None:
    """Process one NWB file.
    
    Args:
        ephys_roi_id: ID of the electrophysiology ROI
        if_save_interpolated: Whether to save interpolated data
        save_dir: Directory to save results
    """
    try:
        # --- 1. Get raw data ---
        raw = PatchSeqNWB(ephys_roi_id=ephys_roi_id)

        # --- 2. Extract features using eFEL ---
        features_dict, raw_traces = extract_features_using_efel(raw, if_save_interpolated)

        # --- 3. Save features_dict to HDF5 using panda's hdf5 store ---
        os.makedirs(f"{save_dir}/features", exist_ok=True)
        save_dict_to_hdf5(features_dict, f"{save_dir}/features/{ephys_roi_id}_efel_features.h5")

        return "Success"
    
    except Exception as e:
        import traceback
        error_message = f"Error processing {ephys_roi_id}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return error_message


if __name__ == "__main__":
    import tqdm

    logging.basicConfig(level=logging.INFO)

    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata

    df_meta = load_ephys_metadata()

    for _ephys_roi_id in ["1418561975"]: #tqdm.tqdm(df_meta["ephys_roi_id_tab_master"][:10]):
        logger.info(f"Processing {_ephys_roi_id}...")
        process_one_nwb(
            ephys_roi_id=str(int(_ephys_roi_id)),
            if_save_interpolated=False,
            save_dir=RESULTS_DIRECTORY
        )
