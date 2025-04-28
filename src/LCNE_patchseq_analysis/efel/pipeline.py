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
from LCNE_patchseq_analysis.pipeline_util.s3 import S3_PATH_BASE, sync_directory

logger = logging.getLogger(__name__)


def extract_efel_features_in_parallel(skip_existing: bool = True, skip_errors: bool = True):
    """Extract eFEL features in parallel."""

    def get_roi_ids():
        df_meta = load_ephys_metadata(combine_roi_ids=True)
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
    os.makedirs(f"{RESULTS_DIRECTORY}/cell_stats", exist_ok=True)
    results = run_parallel_processing(
        process_func=extract_cell_level_stats_one,
        process_func_kwargs={"if_generate_plots": if_generate_plots},
        analysis_name="Extract cell level stats",
        skip_errors=skip_errors,
    )

    # Filter out None results (errors)
    df_cell_stats = pd.concat(
        [result[1]["df_cell_stats"] for result in results if result[0] == "Success"], axis=0
    )
    df_cell_representative_spike_waveforms = pd.concat(
        [
            result[1]["df_cell_representative_spike_waveforms"]
            for result in results
            if result[0] == "Success"
        ],
        axis=0,
    )

    # ---- Merge into Brian's spreadsheet ----
    df_ephys_metadata = load_ephys_metadata(if_from_s3=False, combine_roi_ids=True).rename(
        columns={"ephys_roi_id_tab_master": "ephys_roi_id"}
    )
    df_merged = df_ephys_metadata.merge(df_cell_stats, on="ephys_roi_id", how="left")

    # ---- Post-processing ----
    df_merged = df_merged.rename(
        columns={col: col.replace("_tab_master", "") for col in df_merged.columns},
    )
    df_merged.loc[:, "LC_targeting"] = df_merged["LC_targeting"].fillna("unknown")

    # Remove columns start with efel_
    df_merged = df_merged.loc[:, ~df_merged.columns.str.startswith("efel_")]
    # Remove "first_spike_" in all column names
    df_merged.columns = [col.replace("first_spike_", "") for col in df_merged.columns]
    # Add EFEL_prefix to all columns that has @ in its name
    df_merged.columns = [f"efel_{col}" if "@" in col else col for col in df_merged.columns]

    # ---- Save the summary table to disk ----
    save_path = f"{RESULTS_DIRECTORY}/cell_stats/cell_level_stats.csv"
    df_merged.to_csv(save_path, index=False)

    # ---- Save the representative spike waveforms to disk ----
    save_path = f"{RESULTS_DIRECTORY}/cell_stats/cell_level_spike_waveforms.pkl"
    df_cell_representative_spike_waveforms.to_pickle(save_path)

    logger.info(f"Successfully extracted cell-level stats for {len(df_cell_stats)} cells!")
    logger.info(f"Summary table saved to {save_path}")

    return df_merged


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("-" * 80)
    logger.info("Extracting features in parallel...")
    extract_efel_features_in_parallel(skip_existing=True, skip_errors=True)

    logger.info("-" * 80)
    logger.info("Generating sweep plots in parallel...")
    generate_sweep_plots_in_parallel(skip_existing=True, skip_errors=True)

    logger.info("-" * 80)
    logger.info("Extracting cell-level statistics...")
    extract_cell_level_stats_in_parallel(skip_errors=False, if_generate_plots=True)

    # Sync the whole results directory to S3
    sync_directory(RESULTS_DIRECTORY, S3_PATH_BASE + "/efel", if_copy=False)  # sync only

    # ================================
    # For debugging
    # enerate_sweep_plots_one("1246071525")
