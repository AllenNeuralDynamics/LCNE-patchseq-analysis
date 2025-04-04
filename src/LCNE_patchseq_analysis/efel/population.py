"""
This module contains functions for extracting cell-level statistics from a single eFEL features file.
"""

import logging

from LCNE_patchseq_analysis.efel.io import load_efel_features_from_roi

logger = logging.getLogger(__name__)


def extract_cell_level_stats_one(ephys_roi_id: str):
    """Extract cell-level statistics from a single eFEL features file."""
    try:
        logger.info(f"Extracting cell-level stats for {ephys_roi_id}...")
        # Load features but don't use them yet (placeholder for implementation)
        features_dict = load_efel_features_from_roi(ephys_roi_id)

        df_features_per_sweep = features_dict["df_features_per_sweep"].merge(
            features_dict["df_sweeps"], on="sweep_number"
        )

        # Calculate AP width average for short square rheo stimulus, explicitly excluding NaN values
        AP_width_short_square_rheo_aver = df_features_per_sweep.query(
            "stimulus_code == 'X3LP_Rheo'"
        )["first_spike_AP_duration_half_width"].mean(
            skipna=True
        )  # Explicitly skip NaN values

        # Extract cell-level statistics
        cell_stats = {
            "ephys_roi_id": ephys_roi_id,
            "first_spike_AP_duration_half_width @ short_square_rheo, aver": AP_width_short_square_rheo_aver,
            # Add your cell-level statistics here
            # For example:
            # "mean_firing_rate": features_dict["df_features_per_sweep"]["spike_count"].mean(),
            # "max_ap_width": features_dict["df_features_per_sweep"]["first_spike_AP_width"].max(),
            # etc.
        }

        logger.info(f"Successfully extracted cell-level stats for {ephys_roi_id}!")
        return "Success", cell_stats
    except Exception as e:
        import traceback

        error_message = f"Error processing {ephys_roi_id}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return None


