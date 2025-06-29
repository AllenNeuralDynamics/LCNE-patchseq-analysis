"""
Spike analysis utilities for population analysis.

This module contains functions for extracting and analyzing representative spike
waveforms from electrophysiology data.
"""

from typing import Dict

import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


def normalize_data(x, idx_range_to_norm=None):
    """Normalize data within a specified range."""
    x0 = x if idx_range_to_norm is None else x[:, idx_range_to_norm]
    min_vals = np.min(x0, axis=1, keepdims=True)
    range_vals = np.ptp(x0, axis=1, keepdims=True)
    return (x - min_vals) / range_vals


def normalize_spike_waveform(waveform: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize spike waveform using specified method.

    Parameters:
    -----------
    waveform : np.ndarray
        The spike waveform to normalize
    method : str, default 'minmax'
        Normalization method ('minmax' or 'zscore')

    Returns:
    --------
    np.ndarray
        Normalized waveform
    """
    if method == "minmax":
        min_val = np.min(waveform)
        max_val = np.max(waveform)
        if max_val == min_val:
            return np.zeros_like(waveform)
        return (waveform - min_val) / (max_val - min_val)
    elif method == "zscore":
        mean_val = np.mean(waveform)
        std_val = np.std(waveform)
        if std_val == 0:
            return np.zeros_like(waveform)
        return (waveform - mean_val) / std_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def extract_representative_spikes(
    df_spikes: pd.DataFrame,
    extract_from,
    if_normalize_v: bool = True,
    normalize_window_v: tuple = (-2, 4),
    if_normalize_dvdt: bool = True,
    normalize_window_dvdt: tuple = (-2, 0),
    if_smooth_dvdt: bool = True,
    filtered_df_meta: pd.DataFrame = None,
):
    """Extract and process representative spike waveforms.

    Parameters:
    -----------
    df_spikes : pd.DataFrame
        DataFrame containing spike waveform data
    extract_from : str
        Source to extract spikes from
    if_normalize_v : bool, default=True
        Whether to normalize voltage traces
    normalize_window_v : tuple, default=(-2, 4)
        Time window for voltage normalization
    if_normalize_dvdt : bool, default=True
        Whether to normalize dV/dt traces
    normalize_window_dvdt : tuple, default=(-2, 0)
        Time window for dV/dt normalization
    if_smooth_dvdt : bool, default=True
        Whether to smooth dV/dt traces
    filtered_df_meta : pd.DataFrame, optional
        Metadata filter to apply

    Returns:
    --------
    tuple
        (df_v_norm, df_dvdt_norm) - normalized voltage and dV/dt DataFrames
    """
    # Get the waveforms
    df_waveforms = df_spikes.query("extract_from == @extract_from")

    # Filter by filtered_df_meta
    if filtered_df_meta is not None:
        df_waveforms = df_waveforms.query("ephys_roi_id in @filtered_df_meta.ephys_roi_id.values")

    if len(df_waveforms) == 0:
        raise ValueError(f"No waveforms found for extract_from={extract_from}")

    t = df_waveforms.columns.values.T
    v = df_waveforms.values
    dvdt = np.gradient(v, t, axis=1)

    # Normalize the dvdt
    if if_normalize_dvdt:
        dvdt = normalize_data(
            dvdt,
            idx_range_to_norm=np.where(
                (t >= normalize_window_dvdt[0]) & (t <= normalize_window_dvdt[1])
            )[0],
        )

    if if_smooth_dvdt:
        dvdt = savgol_filter(dvdt, window_length=5, polyorder=3, axis=1)

    dvdt_max_idx = np.argmax(dvdt, axis=1)
    max_shift_right = dvdt_max_idx.max() - dvdt_max_idx.min()

    # Calculate new time array that spans all possible shifts
    dt = t[1] - t[0]
    t_dvdt = -dvdt_max_idx.max() * dt + np.arange(len(t) + max_shift_right) * dt

    # Create new dvdt array with NaN padding
    new_dvdt = np.full((dvdt.shape[0], len(t_dvdt)), np.nan)

    # For each cell, place its dvdt trace in the correct position
    for i, (row, peak_idx) in enumerate(zip(dvdt, dvdt_max_idx)):
        start_idx = dvdt_max_idx.max() - peak_idx  # Align the max_index
        new_dvdt[i, start_idx : start_idx + len(row)] = row

    # Normalize the v
    if if_normalize_v:
        idx_range_to_norm = np.where((t >= normalize_window_v[0]) & (t <= normalize_window_v[1]))[0]
        v = normalize_data(v, idx_range_to_norm)

    # Create dictionary with ephys_roi_id as keys
    df_v_norm = pd.DataFrame(v, index=df_waveforms.index.get_level_values(0), columns=t)
    df_dvdt_norm = pd.DataFrame(
        new_dvdt, index=df_waveforms.index.get_level_values(0), columns=t_dvdt
    )

    return df_v_norm, df_dvdt_norm


def extract_simple_representative_spikes(
    df_spikes: pd.DataFrame, normalization_method: str = "minmax"
) -> Dict[str, Dict]:
    """
    Extract simple representative spike waveforms for each cell.

    This is a simpler version that calculates mean waveforms per cell.

    Parameters:
    -----------
    df_spikes : pd.DataFrame
        DataFrame containing spike waveforms with columns:
        - 'cell_specimen_id': cell identifier
        - 'waveform': spike waveform data
    normalization_method : str, default 'minmax'
        Method for normalizing waveforms ('minmax' or 'zscore')

    Returns:
    --------
    Dict[str, Dict]
        Dictionary mapping cell_specimen_id to spike data containing:
        - 'waveform': representative waveform
        - 'normalized_waveform': normalized representative waveform
        - 'n_spikes': number of spikes used
    """
    representative_spikes = {}

    for cell_id in df_spikes["cell_specimen_id"].unique():
        cell_spikes = df_spikes[df_spikes["cell_specimen_id"] == cell_id]

        if len(cell_spikes) == 0:
            continue

        # Extract waveforms
        waveforms = []
        for _, row in cell_spikes.iterrows():
            waveform = row["waveform"]
            if isinstance(waveform, (list, np.ndarray)) and len(waveform) > 0:
                waveforms.append(np.array(waveform))

        if not waveforms:
            continue

        # Convert to numpy array for easier manipulation
        waveforms = np.array(waveforms)

        # Calculate representative waveform (mean)
        representative_waveform = np.mean(waveforms, axis=0)

        # Normalize the representative waveform
        normalized_waveform = normalize_spike_waveform(
            representative_waveform, method=normalization_method
        )

        representative_spikes[str(cell_id)] = {
            "waveform": representative_waveform,
            "normalized_waveform": normalized_waveform,
            "n_spikes": len(waveforms),
        }

    return representative_spikes
