"""eFEL pipeline."""

import glob
import json
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


def extract_efel_features_in_parallel(skip_existing: bool = True, skip_errors: bool = True):
    """Extract eFEL features in parallel."""
    pool = mp.Pool(processes=mp.cpu_count())
    df_meta = load_ephys_metadata()
    all_ephys_roi_ids = df_meta["ephys_roi_id_tab_master"]

    if skip_existing:
        # Exclude ROI IDs that already have eFEL features
        all_ephys_roi_ids = [
            eph for eph in all_ephys_roi_ids if not os.path.exists(
                f"{RESULTS_DIRECTORY}/features/{int(eph)}_efel.h5"
            )
        ]
        n_skipped_existing = len(df_meta) - len(all_ephys_roi_ids)
        
    n_skipped_errors = 0
    if skip_errors:
        # Exclude ROI IDs that have errors
        error_file = f"{RESULTS_DIRECTORY}/pipeline_error_Extract eFEL features.json"
        if os.path.exists(error_file):
            with open(error_file, "r") as f:
                errors_list = json.load(f)
            len_before = len(all_ephys_roi_ids)
            all_ephys_roi_ids = [
                eph for eph in all_ephys_roi_ids if not any(eph == error["roi_id"] for error in errors_list)
            ]
            n_skipped_errors = len_before - len(all_ephys_roi_ids)

    with pool:
        # Queue all tasks
        jobs = []
        
        # # For debugging
        # all_ephys_roi_ids = ["1408379728", '1239045986', '1298901226', '1408386647', '1393880949', 
        #              '1321654119', '1401854370', '1314005711', '1246391218', '1406415579']

        for _ephys_roi_id in all_ephys_roi_ids:
            job = pool.apply_async(
                extract_efel_one, args=(_ephys_roi_id, False, RESULTS_DIRECTORY)
            )
            jobs.append(job)

        # Wait for all processes to complete
        results = [job.get() for job in tqdm(jobs)]

        handle_errors(results, all_ephys_roi_ids, "Extract eFEL features")
        if skip_existing:
            logger.info(f"Skipped {n_skipped_existing} ROI IDs that already have eFEL features")
        if skip_errors:
            logger.info(f"Skipped {n_skipped_errors} ROI IDs that had errors before")

    return results


def handle_errors(results, roi_ids, analysis_name: str):
    # Show how many successful and failed processes
    errors = [
        {"roi_id": roi_ids[i], "error": result}
        for i, result in enumerate(results)
        if result != "Success"
    ]

    logger.info(f"{analysis_name}, Success: {len(results) - len(errors)}")
    if len(errors) > 0:
        logger.error(f"{analysis_name}, Failed: {len(errors)}")

    # Append erros to the list in json
    error_file = f"{RESULTS_DIRECTORY}/pipeline_error_{analysis_name}.json"
    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            errors_list = json.load(f)
    else:
        errors_list = []
    with open(error_file, "w") as f:
        json.dump(errors_list + errors, f, indent=4)
    return


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
    pool = mp.Pool(processes=mp.cpu_count())

    # Find all h5 under RESULTS_DIRECTORY/features
    feature_h5_files = glob.glob(f"{RESULTS_DIRECTORY}/features/*.h5")
    ephys_roi_ids = [
        os.path.basename(feature_h5_file).split("_")[0] for feature_h5_file in feature_h5_files
    ]

    if skip_existing:
        # Exclude ROI IDs that already have ALL success sweep plots
        ephys_roi_ids = [
            eph for eph in ephys_roi_ids if not os.path.exists(
                f"{RESULTS_DIRECTORY}/plots/{int(eph)}/all_success"
            )
        ]
        n_skipped_existing = len(feature_h5_files) - len(ephys_roi_ids)

    n_skipped_errors = 0
    if skip_errors:
        error_file = f"{RESULTS_DIRECTORY}/pipeline_error_Generate sweep plots.json"
        if os.path.exists(error_file):
            # Exclude ROI IDs that have errors
            with open(error_file, "r") as f:
                errors_list = json.load(f)
            len_before = len(ephys_roi_ids)
            ephys_roi_ids = [
                eph for eph in ephys_roi_ids if not any(eph == error["roi_id"] for error in errors_list)
            ]
            n_skipped_errors = len_before - len(ephys_roi_ids)

    # Queue all tasks
    # For debugging
    # ephys_roi_ids = ['1246391218', '1406415579', '1394153301', '1403961154', '1266606241', '1246071525']
    
    jobs = []
    for ephys_roi_id in ephys_roi_ids:
        job = pool.apply_async(generate_sweep_plots_one, args=(ephys_roi_id,))
        jobs.append(job)

    # Wait for all processes to complete
    results = [job.get() for job in tqdm(jobs)]

    handle_errors(results, ephys_roi_ids, "Generate sweep plots")
    
    if skip_existing:
        logger.info(f"Skipped {n_skipped_existing} ROI IDs that already have sweep plots")
    if skip_errors:
        logger.info(f"Skipped {n_skipped_errors} ROI IDs that had errors before")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("-" * 80)
    logger.info("Extracting features in parallel...")
    extract_efel_features_in_parallel(skip_existing=True, skip_errors=True)

    logger.info("-" * 80)
    logger.info("Generating sweep plots in parallel...")
    generate_sweep_plots_in_parallel(skip_existing=True, skip_errors=True)
    
    # For debugging
    # enerate_sweep_plots_one("1246071525")
