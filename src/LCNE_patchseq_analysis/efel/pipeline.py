"""eFEL pipeline."""

import logging
import os

from LCNE_patchseq_analysis import RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.efel.core import extract_efel_one
from LCNE_patchseq_analysis.efel.io import load_efel_features_from_roi
from LCNE_patchseq_analysis.efel.plot import plot_sweep_summary
from LCNE_patchseq_analysis.efel.util import run_parallel_processing

logger = logging.getLogger(__name__)


def extract_efel_features_in_parallel(skip_existing: bool = True, skip_errors: bool = True):
    """Extract eFEL features in parallel."""

    def get_roi_ids():
        df_meta = load_ephys_metadata()
        return df_meta["ephys_roi_id_tab_master"]

    def check_existing(ephys_roi_id):
        return os.path.exists(f"{RESULTS_DIRECTORY}/features/{int(ephys_roi_id)}_efel.h5")

    return run_parallel_processing(
        process_func=extract_efel_one,
        analysis_name="Extract eFEL features",
        get_roi_ids_func=get_roi_ids,
        skip_existing=skip_existing,
        skip_errors=skip_errors,
        existing_check_func=check_existing,
    )


def generate_sweep_plots_one(ephys_roi_id: str):
    """Load from HDF5 file and generate sweep plots in parallel."""
    try:
        logger.info(f"Generating sweep plots for {ephys_roi_id}...")
        features_dict = load_efel_features_from_roi(ephys_roi_id)
        plot_sweep_summary(features_dict, f"{RESULTS_DIRECTORY}/plots")
        logger.info(f"Successfully generated sweep plots for {ephys_roi_id}!")
        return "Success"
    except Exception as e:
        import traceback

        error_message = f"Error processing {ephys_roi_id}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return error_message


def generate_sweep_plots_in_parallel(skip_existing: bool = True, skip_errors: bool = True):
    """Generate sweep plots in parallel."""

    def check_existing(ephys_roi_id):
        return os.path.exists(f"{RESULTS_DIRECTORY}/plots/{int(ephys_roi_id)}/all_success")

    return run_parallel_processing(
        process_func=generate_sweep_plots_one,
        analysis_name="Generate sweep plots",
        skip_existing=skip_existing,
        skip_errors=skip_errors,
        existing_check_func=check_existing,
    )


def extract_cell_level_stats_one(ephys_roi_id: str):
    """Extract cell-level statistics from a single eFEL features file."""
    try:
        logger.info(f"Extracting cell-level stats for {ephys_roi_id}...")
        # Load features but don't use them yet (placeholder for implementation)
        load_efel_features_from_roi(ephys_roi_id)

        # Extract cell-level statistics
        # This is a placeholder for the actual implementation
        # You will need to fill in the specific statistics you want to extract
        cell_stats = {
            "ephys_roi_id": ephys_roi_id,
            # Add your cell-level statistics here
            # For example:
            # "mean_firing_rate": features_dict["df_features_per_sweep"]["spike_count"].mean(),
            # "max_ap_width": features_dict["df_features_per_sweep"]["first_spike_AP_width"].max(),
            # etc.
        }

        logger.info(f"Successfully extracted cell-level stats for {ephys_roi_id}!")
        return cell_stats
    except Exception as e:
        import traceback

        error_message = f"Error processing {ephys_roi_id}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return None


def extract_cell_level_stats_in_parallel(skip_errors: bool = True):
    """Extract cell-level statistics from all available eFEL features files in parallel."""

    def post_process(results):
        # Filter out None results (errors)
        valid_results = [result for result in results if result is not None]

        # Create a summary table
        import pandas as pd

        df_cell_stats = pd.DataFrame(valid_results)

        # Save the summary table to disk
        os.makedirs(f"{RESULTS_DIRECTORY}/cell_stats", exist_ok=True)
        df_cell_stats.to_csv(f"{RESULTS_DIRECTORY}/cell_stats/cell_level_stats.csv", index=False)

        logger.info(f"Successfully extracted cell-level stats for {len(valid_results)} cells")

        return df_cell_stats

    return run_parallel_processing(
        process_func=extract_cell_level_stats_one,
        analysis_name="Extract cell level stats",
        skip_errors=skip_errors,
        post_process_func=post_process,
    )


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("-" * 80)
    logger.info("Extracting features in parallel...")
    extract_efel_features_in_parallel(skip_existing=True, skip_errors=True)

    logger.info("-" * 80)
    logger.info("Generating sweep plots in parallel...")
    generate_sweep_plots_in_parallel(skip_existing=True, skip_errors=True)

    # logger.info("-" * 80)
    # logger.info("Extracting cell-level statistics...")
    # extract_cell_level_stats_in_parallel(skip_errors=True)

    # For debugging
    # enerate_sweep_plots_one("1246071525")
