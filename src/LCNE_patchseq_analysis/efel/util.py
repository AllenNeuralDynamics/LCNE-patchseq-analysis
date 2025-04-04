"""Utilities for eFEL analysis."""

import glob
import json
import logging
import multiprocessing as mp
import os
from typing import Any, Callable, List, Optional

from tqdm import tqdm

from LCNE_patchseq_analysis import RESULTS_DIRECTORY

logger = logging.getLogger(__name__)


def run_parallel_processing(
    process_func: Callable,
    analysis_name: str,
    get_roi_ids_func: Optional[Callable] = None,
    skip_existing: bool = True,
    skip_errors: bool = True,
    existing_check_func: Optional[Callable] = None,
) -> List[Any]:
    """
    Generic function to run parallel processing tasks.

    Args:
        process_func: Function to process a single ROI ID
        analysis_name: Name of the analysis for error handling
        get_roi_ids_func: Function to get ROI IDs (if None, will use all .h5 files)
        skip_existing: Whether to skip existing results
        skip_errors: Whether to skip ROIs with previous errors
        existing_check_func: Function to check if results already exist

    Returns:
        List of results from the parallel processing
    """
    pool = mp.Pool(processes=mp.cpu_count())

    # Get ROI IDs
    if get_roi_ids_func:
        ephys_roi_ids = get_roi_ids_func()
    else:
        # Default: find all h5 under RESULTS_DIRECTORY/features
        feature_h5_files = glob.glob(f"{RESULTS_DIRECTORY}/features/*.h5")
        ephys_roi_ids = [
            os.path.basename(feature_h5_file).split("_")[0] for feature_h5_file in feature_h5_files
        ]

    n_skipped_existing = 0
    if skip_existing and existing_check_func:
        # Exclude ROI IDs that already have results
        len_before = len(ephys_roi_ids)
        ephys_roi_ids = [eph for eph in ephys_roi_ids if not existing_check_func(eph)]
        n_skipped_existing = len_before - len(ephys_roi_ids)

    n_skipped_errors = 0
    if skip_errors:
        # Exclude ROI IDs that have errors
        error_file = f"{RESULTS_DIRECTORY}/pipeline_error_{analysis_name}.json"
        if os.path.exists(error_file):
            with open(error_file, "r") as f:
                errors_list = json.load(f)
            len_before = len(ephys_roi_ids)
            ephys_roi_ids = [
                eph
                for eph in ephys_roi_ids
                if not any(eph == error["roi_id"] for error in errors_list)
            ]
            n_skipped_errors = len_before - len(ephys_roi_ids)

    # Queue all tasks
    jobs = []
    for ephys_roi_id in ephys_roi_ids:
        job = pool.apply_async(process_func, args=(ephys_roi_id,))
        jobs.append(job)

    # Wait for all processes to complete
    results = [job.get() for job in tqdm(jobs, desc=f"Processing {analysis_name}")]

    # Handle errors
    handle_errors(results, ephys_roi_ids, analysis_name)

    # Log skipped items
    if skip_existing:
        logger.info(f"Skipped {n_skipped_existing} ROI IDs that already have results")
    if skip_errors:
        logger.info(f"Skipped {n_skipped_errors} ROI IDs that had errors before")

    return results


def handle_errors(results, roi_ids, analysis_name: str):
    """
    Handle errors from parallel processing.

    Args:
        results: List of results from parallel processing
        roi_ids: List of ROI IDs
        analysis_name: Name of the analysis
    """
    # Show how many successful and failed processes
    errors = [
        {"roi_id": roi_ids[i], "error": result}
        for i, result in enumerate(results)
        if result != "Success" and result[0] != "Success"  # Result is a tuple
    ]

    logger.info(f"{analysis_name}, Success: {len(results) - len(errors)}")
    if len(errors) > 0:
        logger.error(f"{analysis_name}, Failed: {len(errors)}")

    # Append errors to the list in json
    error_file = f"{RESULTS_DIRECTORY}/pipeline_error_{analysis_name}.json"
    if os.path.exists(error_file):
        with open(error_file, "r") as f:
            errors_list = json.load(f)
    else:
        errors_list = []
    with open(error_file, "w") as f:
        json.dump(errors_list + errors, f, indent=4)
    return
