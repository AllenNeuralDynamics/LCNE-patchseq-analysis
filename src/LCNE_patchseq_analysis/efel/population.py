"""
This module contains functions for extracting cell-level statistics
from a single eFEL features file.
"""

import logging
from typing import Literal

import pandas as pd

from LCNE_patchseq_analysis.efel import (
    EXTRACT_SPIKE_FROMS, EXTRACT_SAG_FROMS,
    EXTRACT_SPIKE_FEATURES, EXTRACT_SAG_FEATURES
)
from LCNE_patchseq_analysis.efel.io import load_efel_features_from_roi

logger = logging.getLogger(__name__)



def df_sweep_selector(df: pd.DataFrame,
                   stim_type: Literal["subthreshold", "short_square_rheo",
                                      "long_square_rheo", "long_square_supra"],
                   aggregate_method: Literal["min", "aver"] | int) -> pd.DataFrame:
    """Select sweeps based on stimulus type and aggregation method."""
    
    def _get_min_or_aver(df_this, aggregate_method):
        if aggregate_method == "aver":
            return df_this
        elif aggregate_method == "min":
            # Find the sweep with the minimum stimulus amplitude that has at least 1 spike
            min_idx = df_this["stimulus_amplitude"].abs().idxmin()
            return df_this.loc[[min_idx]]
        else:
            raise ValueError("aggregate_method must be 'aver' or 'min'")
    
    if stim_type == "subthreshold":
        if aggregate_method == "aver":
            # All SubThreshold sweeps
            return df.query(
                "stimulus_code.str.contains('SubThresh')"
                "and stimulus_name == 'Long Square'"
            )
        elif isinstance(aggregate_method, int):
            return df.query(
                "stimulus_code.str.contains('SubThresh')"
                " and stimulus_name == 'Long Square'"
                # Allow for 1 mV tolerance
                " and abs(abs(stimulus_amplitude) - abs(@aggregate_method)) < 1"
            )
        else:
            raise ValueError("aggregate_method must be 'aver' or an integer (the abs(amplitude)"
                             "of the stimulus) for subthreshold sweeps")
    elif stim_type == "short_square_rheo":
        df_this = df.query(
            "stimulus_code.str.contains('Rheo')"
            "and stimulus_name == 'Short Square'"
            "and spike_count > 0"
        )
    elif stim_type == "long_square_rheo":
        df_this = df.query(
            "stimulus_code.str.contains('Rheo')"
            "and stimulus_name == 'Long Square'"
            "and spike_count > 0"
        )
    elif stim_type == "long_square_supra":
        df_this = df.query(
            "stimulus_code.str.contains('SupraThresh')"
            "and stimulus_name == 'Long Square'"
            "and spike_count > 0"
        )
    else:
        raise ValueError(f"Invalid stimulus type: {stim_type}")

    if df_this.empty:
        return None
    return _get_min_or_aver(df_this, aggregate_method)


def extract_cell_level_stats_one(ephys_roi_id: str):
    """Extract cell-level statistics from a single eFEL features file."""
    try:
        logger.info(f"Extracting cell-level stats for {ephys_roi_id}...")
        # Load features but don't use them yet (placeholder for implementation)
        features_dict = load_efel_features_from_roi(ephys_roi_id)

        df_features_per_sweep = features_dict["df_features_per_sweep"].merge(
            features_dict["df_sweeps"], on="sweep_number"
        )

        cell_stats_dict = {}

        # Loop over spike and sag features
        for feature_type, features_to_extract in [
            (EXTRACT_SPIKE_FROMS, EXTRACT_SPIKE_FEATURES),
            (EXTRACT_SAG_FROMS, EXTRACT_SAG_FEATURES),
        ]:
            for key, value in feature_type.items():
                df_sweep = df_sweep_selector(
                    df_features_per_sweep, stim_type=value[0], aggregate_method=value[1])
                if df_sweep is not None:
                    # Calculate mean over rows for each feature
                    mean_values = df_sweep[features_to_extract].mean()
                    # Create a dictionary with feature names and their mean values
                    feature_dict = {f"{feature} @ {key}": value for feature, value in mean_values.items()}
                    cell_stats_dict.update(feature_dict)

        cell_stats = pd.DataFrame(cell_stats_dict, index=pd.Index([ephys_roi_id], name="ephys_roi_id"))

        logger.info(f"Successfully extracted cell-level stats for {ephys_roi_id}!")
        return "Success", cell_stats
    except Exception as e:
        import traceback

        error_message = f"Error processing {ephys_roi_id}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return None



if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    status, cell_stats = extract_cell_level_stats_one("1212557784")
    cell_stats.to_csv("./cell_stats.csv")