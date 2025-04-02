"""eFEL pipeline."""

import logging
import multiprocessing as mp
from tqdm import tqdm

from LCNE_patchseq_analysis import RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.efel.io import load_dict_from_hdf5
from LCNE_patchseq_analysis.efel.single_cell import process_one_nwb
from LCNE_patchseq_analysis.efel.plot import plot_sweep_summary
logger = logging.getLogger(__name__)


def process_all_nwbs_in_parallel():
    """Process all NWBs in parallel."""
    pool = mp.Pool(processes=mp.cpu_count())
    df_meta = load_ephys_metadata()
    all_ephys_roi_ids = df_meta["ephys_roi_id_tab_master"]
    
    with pool:
        # Queue all tasks
        jobs = []
        for _ephys_roi_id in all_ephys_roi_ids[:10]:
            job = pool.apply_async(
                process_one_nwb,
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


def generate_sweep_plots_one(ephys_roi_id: str):
    """Load from HDF5 file and generate sweep plots in parallel."""

    features_dict = load_dict_from_hdf5(
        f"{RESULTS_DIRECTORY}/features/{ephys_roi_id}_efel_features.h5")
    plot_sweep_summary(features_dict, f"{RESULTS_DIRECTORY}/plots/{ephys_roi_id}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # logger.info("-" * 80)
    # logger.info("Processing all NWBs in parallel...")
    # process_all_nwbs_in_parallel()
    
    logger.info("-" * 80)
    logger.info("Generating sweep plots in parallel...")
    
    generate_sweep_plots_one("1418561975")
