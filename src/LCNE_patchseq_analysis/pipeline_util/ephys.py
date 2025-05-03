"""Ephys-related data utils"""

import concurrent.futures
import logging
import os

import pandas as pd
from tqdm import tqdm

from LCNE_patchseq_analysis.pipeline_util.metadata import read_brian_spreadsheet
from LCNE_patchseq_analysis.pipeline_util.s3 import sync_directory

logger = logging.getLogger(__name__)

s3_bucket = "s3://aind-scratch-data/aind-patchseq-data/raw"


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

def trigger_patchseq_upload(
    metadata_path=os.path.expanduser(R"~\Downloads\IVSCC_LC_summary.xlsx"),
    upload_raw_data=True,
):
    # Generate a list of isilon paths
    dfs = read_brian_spreadsheet(file_path=metadata_path, add_lims=True)
    df_merged = dfs["df_merged"]

    # Also save df_merged as csv and upload to s3
    df_merged.to_csv("df_metadata_merged.csv", index=False)

    # Conditionally upload raw data
    if upload_raw_data:
        logger.info("Uploading raw data to S3...")
        upload_raw_from_isilon_to_s3_batch(df_merged, s3_bucket=s3_bucket, max_workers=10)

    logger.info("Uploading df_metadata_merged.csv to S3...")
    sync_directory("df_metadata_merged.csv", s3_bucket + "/df_metadata_merged.csv", if_copy=True)


if __name__ == "__main__":

    # Set logger level
    logging.basicConfig(level=logging.DEBUG)

    trigger_patchseq_upload(
        os.path.expanduser(R"~\Downloads\IVSCC_LC_summary.xlsx"), upload_raw_data=False
    )
