"""eFEL pipeline."""

import logging
import os

import pandas as pd

from LCNE_patchseq_analysis import RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.efel.core import extract_efel_one
from LCNE_patchseq_analysis.efel.plot import generate_sweep_plots_one
from LCNE_patchseq_analysis.efel.population import extract_cell_level_stats_one
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


def extract_cell_level_stats_in_parallel(skip_errors: bool = True, if_generate_plots: bool = True):
    """Extract cell-level statistics from all available eFEL features files in parallel."""

    # ---- Extract cell-level stats ----
    results = run_parallel_processing(
        process_func=extract_cell_level_stats_one,
        process_func_kwargs={"if_generate_plots": if_generate_plots},
        analysis_name="Extract cell level stats",
        skip_errors=skip_errors,
    )

    # Filter out None results (errors)
    valid_results = [result[1] for result in results if result is not None]

    df_cell_stats = pd.concat(valid_results, axis=0)

    # ---- Merge into Brian's spreadsheet ----
    df_ephys_metadata = load_ephys_metadata(if_with_efel=False).rename(
        columns={"ephys_roi_id_tab_master": "ephys_roi_id"}
    )
    df_merged = df_ephys_metadata.merge(df_cell_stats, on="ephys_roi_id", how="left")

    # ---- Save the summary table to disk ----
    os.makedirs(f"{RESULTS_DIRECTORY}/cell_stats", exist_ok=True)
    save_path = f"{RESULTS_DIRECTORY}/cell_stats/cell_level_stats.csv"
    df_merged.to_csv(save_path, index=False)

    logger.info(f"Successfully extracted cell-level stats for {len(valid_results)} cells!")
    logger.info(f"Summary table saved to {save_path}")

    return df_merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # logger.info("-" * 80)
    # logger.info("Extracting features in parallel...")
    # extract_efel_features_in_parallel(skip_existing=True, skip_errors=True)

    # logger.info("-" * 80)
    # logger.info("Generating sweep plots in parallel...")
    # generate_sweep_plots_in_parallel(skip_existing=True, skip_errors=True)

    logger.info("-" * 80)
    logger.info("Extracting cell-level statistics...")
    extract_cell_level_stats_in_parallel(skip_errors=False, if_generate_plots=True)

    # ================================
    # For debugging
    # enerate_sweep_plots_one("1246071525")
