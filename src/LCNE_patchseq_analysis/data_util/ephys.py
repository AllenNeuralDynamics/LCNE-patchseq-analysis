"""Ephys-related data utils"""

import os
import subprocess
import logging
import concurrent.futures
from tqdm import tqdm

import pandas as pd

from LCNE_patchseq_analysis.data_util.metadata import read_brian_spreadsheet

logger = logging.getLogger(__name__)

s3_bucket = "s3://aind-scratch-data/aind-patchseq-data/raw"


def sync_directory(local_dir, destination):
    """
    Sync the local directory with the given S3 destination using aws s3 sync.
    Returns a status string based on the command output.
    """
    try:
        # Run aws s3 sync command and capture the output
        result = subprocess.run(
            ["aws", "s3", "sync", local_dir, destination], capture_output=True, text=True
        )
        output = result.stdout + result.stderr

        # Check output: if "upload:" appears, files were sent;
        # otherwise, assume that nothing needed uploading.
        if "upload:" in output:
            logger.info(f'Uploaded {local_dir} to {destination}!')
            return "successfully uploaded"
        else:
            logger.info(f'Already exists, skip {local_dir}.')
            return "already exists, skip"
    except Exception as e:
        return f"error during sync: {e}"


def upload_one(row, s3_bucket):
    """Process a single row: normalize the path, check existence,
    and perform (or simulate) the sync.
    """
    # Check if the storage_directory_combined value is null.
    if pd.isnull(row["storage_directory_combined"]):
        logger.info("The path is null")
        status = "the path is null"
        path = None
    else:
        # Normalize the path and prepend a backslash.
        path = "\\" + os.path.normpath(row["storage_directory_combined"])
        roi_name = os.path.basename(path)

        # Check if the local path exists.
        if not os.path.exists(path):
            logger.info(f"Cannot find the path: {path}")
            status = "cannot find the path"
        else:
            logger.info(f"Syncing {path} to {s3_bucket}/{roi_name}...")
            status = sync_directory(path, s3_bucket + "/" + roi_name)
    return {"storage_directory": path, "status": status}


def upload_raw_from_isilon_to_s3_batch(df, s3_bucket=s3_bucket, max_workers=10):
    """Upload raw data from Isilon to S3, using the metadata dataframe in parallel."""
    results = []

    # Create a thread pool to process rows in parallel.
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        # Submit each row for processing.
        futures = [executor.submit(upload_one, row, s3_bucket) for idx, row in df.iterrows()]

        # Collect the results as they complete.
        for future in tqdm(
            concurrent.futures.as_completed(futures), total=len(futures), desc="Uploading..."
        ):
            results.append(future.result())

    logger.info(f"Uploaded {len(results)} files to {s3_bucket} in parallel...")
    logger.info(
        f'Successful uploads: {len([r for r in results if r["status"] == "successfully uploaded"])}'
    )
    logger.info(f'Skiped: {len([r for r in results if r["status"] == "already exists, skip"])}')
    logger.info(
        f'Error during sync: {len([r for r in results if r["status"] == "error during sync"])}'
    )
    logger.info(
        "Cannot find on Isilon: "
        f'{len([r for r in results if r["status"] == "cannot find the path"])}'
    )
    logger.info(f'Null path: {len([r for r in results if r["status"] == "the path is null"])}')

    return pd.DataFrame(results)


if __name__ == "__main__":

    # Set logger level
    logging.basicConfig(level=logging.INFO)

    # Generate a list of isilon paths
    dfs = read_brian_spreadsheet(add_lims=True)

    upload_raw_from_isilon_to_s3_batch(dfs["df_all"].iloc[:10], s3_bucket=s3_bucket, max_workers=10)
