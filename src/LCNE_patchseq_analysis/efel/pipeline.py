"""eFEL pipeline."""

import glob
import logging
import multiprocessing as mp
import os

from tqdm import tqdm

from LCNE_patchseq_analysis import RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.efel.core import extract_efel_one
from LCNE_patchseq_analysis.efel.io import load_efel_features_from_roi
from LCNE_patchseq_analysis.efel.plot import plot_sweep_summary

logger = logging.getLogger(__name__)


def extract_efel_features_in_parallel(only_new: bool = True):
    """Extract eFEL features in parallel."""
    pool = mp.Pool(processes=mp.cpu_count())
    df_meta = load_ephys_metadata()
    all_ephys_roi_ids = df_meta["ephys_roi_id_tab_master"]

    if only_new:
        # Exclude ROI IDs that already have eFEL features
        all_ephys_roi_ids = [
            eph for eph in all_ephys_roi_ids if not os.path.exists(
                f"{RESULTS_DIRECTORY}/features/{int(eph)}_efel.h5"
            )
        ]
        n_skipped = len(df_meta) - len(all_ephys_roi_ids)

    with pool:
        # Queue all tasks
        jobs = []
        for _ephys_roi_id in all_ephys_roi_ids:
            job = pool.apply_async(
                extract_efel_one, args=(str(int(_ephys_roi_id)), False, RESULTS_DIRECTORY)
            )
            jobs.append(job)

        # Wait for all processes to complete
        results = [job.get() for job in tqdm(jobs)]

    # Show how many successful and failed processes
    error_roi_ids = [
        all_ephys_roi_ids[i] for i, result in enumerate(results) if result != "Success"
    ]
    if len(error_roi_ids) > 0:
        logger.error(f"Failed processes: {len(error_roi_ids)}")
        logger.error(f"Failed ROI IDs: {error_roi_ids}")
    logger.info(f"Successful processes: {len(results) - len(error_roi_ids)}")
    if only_new:
        logger.info(f"Skipped {n_skipped} ROI IDs that already have eFEL features")

    return results


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


def generate_sweep_plots_in_parallel(only_new: bool = True):
    """Generate sweep plots in parallel."""
    pool = mp.Pool(processes=mp.cpu_count())

    # Find all h5 under RESULTS_DIRECTORY/features
    feature_h5_files = glob.glob(f"{RESULTS_DIRECTORY}/features/*.h5")
    ephys_roi_ids = [
        os.path.basename(feature_h5_file).split("_")[0] for feature_h5_file in feature_h5_files
    ]

    if only_new:
        # Exclude ROI IDs that already have ALL success sweep plots
        ephys_roi_ids = [
            eph for eph in ephys_roi_ids if not os.path.exists(
                f"{RESULTS_DIRECTORY}/plots/{int(eph)}/all_success"
            )
        ]
        n_skipped = len(feature_h5_files) - len(ephys_roi_ids)

    # Queue all tasks
    jobs = []
    for ephys_roi_id in ephys_roi_ids:
        job = pool.apply_async(generate_sweep_plots_one, args=(ephys_roi_id,))
        jobs.append(job)

    # Wait for all processes to complete
    results = [job.get() for job in tqdm(jobs)]

    # Show how many successful and failed processes
    error_roi_ids = [ephys_roi_ids[i] for i, result in enumerate(results) if result != "Success"]
    if len(error_roi_ids) > 0:
        logger.error(f"Failed processes: {len(error_roi_ids)}")
        logger.error(f"Failed ROI IDs: {error_roi_ids}")
    logger.info(f"Successful processes: {len(results) - len(error_roi_ids)}")
    if only_new:
        logger.info(f"Skipped {n_skipped} ROI IDs that already have sweep plots")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("-" * 80)
    logger.info("Extracting features in parallel...")
    extract_efel_features_in_parallel(only_new=True)

    logger.info("-" * 80)
    logger.info("Generating sweep plots in parallel...")
    generate_sweep_plots_in_parallel(only_new=True)
