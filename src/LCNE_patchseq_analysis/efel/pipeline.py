"""eFEL pipeline."""

import logging
import multiprocessing as mp
import os
import glob
from tqdm import tqdm

from LCNE_patchseq_analysis import RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.efel.io import load_dict_from_hdf5
from LCNE_patchseq_analysis.efel.core import extract_efel_one
from LCNE_patchseq_analysis.efel.plot import plot_sweep_summary
logger = logging.getLogger(__name__)


def extract_efel_features_in_parallel():
    """Extract eFEL features in parallel."""
    pool = mp.Pool(processes=mp.cpu_count())
    df_meta = load_ephys_metadata()
    all_ephys_roi_ids = df_meta["ephys_roi_id_tab_master"][:10]
    
    with pool:
        # Queue all tasks
        jobs = []
        for _ephys_roi_id in all_ephys_roi_ids:
            job = pool.apply_async(
                extract_efel_one,
                args=(str(int(_ephys_roi_id)), False, RESULTS_DIRECTORY)
            )
            jobs.append(job)
        
        # Wait for all processes to complete
        results = [job.get() for job in tqdm(jobs)]

    # Show how many successful and failed processes
    error_roi_ids = [all_ephys_roi_ids[i] for i, result in enumerate(results) if result != "Success"]
    if len(error_roi_ids) > 0:
        logger.error(f"Failed processes: {len(error_roi_ids)}")
        logger.error(f"Failed ROI IDs: {error_roi_ids}")
    logger.info(f"Successful processes: {len(results) - len(error_roi_ids)}")

    return results


def generate_sweep_plots_one(feature_h5_file: str):
    """Load from HDF5 file and generate sweep plots in parallel."""
    ephys_roi_id = os.path.basename(feature_h5_file).split("_")[0]
    try:
        features_dict = load_dict_from_hdf5(feature_h5_file)
        plot_sweep_summary(features_dict, f"{RESULTS_DIRECTORY}/plots")
        return "Success"
    except Exception as e:
        import traceback
        error_message = f"Error processing {ephys_roi_id}: {str(e)}\n{traceback.format_exc()}"
        logger.error(error_message)
        return error_message
    

def generate_sweep_plots_in_parallel():
    """Generate sweep plots in parallel."""
    pool = mp.Pool(processes=mp.cpu_count())
    
    # Find all h5 under RESULTS_DIRECTORY/features
    feature_h5_files = glob.glob(f"{RESULTS_DIRECTORY}/features/*.h5")
    ephys_roi_ids = [os.path.basename(feature_h5_file).split("_")[0] for feature_h5_file in feature_h5_files]
    
    # Queue all tasks
    jobs = []
    for feature_h5_file in feature_h5_files:
        job = pool.apply_async(generate_sweep_plots_one, args=(feature_h5_file,))
        jobs.append(job)
    
    # Wait for all processes to complete
    results = [job.get() for job in tqdm(jobs)]
    
    # Show how many successful and failed processes
    error_roi_ids = [ephys_roi_ids[i] for i, result in enumerate(results) if result != "Success"]
    if len(error_roi_ids) > 0:
        logger.error(f"Failed processes: {len(error_roi_ids)}")
        logger.error(f"Failed ROI IDs: {error_roi_ids}")
    logger.info(f"Successful processes: {len(results) - len(error_roi_ids)}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    logger.info("-" * 80)
    logger.info("Extracting features in parallel...")
    extract_efel_features_in_parallel()
    
    logger.info("-" * 80)
    logger.info("Generating sweep plots in parallel...")
    generate_sweep_plots_in_parallel()
