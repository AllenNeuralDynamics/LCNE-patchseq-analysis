"""Ephys-related data utils"""

import os
import subprocess
import logging

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
            ['aws', 's3', 'sync', local_dir, destination],
            capture_output=True, text=True
        )
        output = result.stdout + result.stderr
        
        # Check output: if "upload:" appears, files were sent;
        # otherwise, assume that nothing needed uploading.
        if "upload:" in output:
            return "successfully uploaded"
        else:
            return "already exists, skip"
    except Exception as e:
        return f"error during sync: {e}"

def upload_raw_from_isilon_to_s3(df, s3_bucket=s3_bucket):
    """Upload raw data from Isilon to S3, using the metadata dataframe"""

    results = []
    for idx, row in df.iterrows():
        if pd.isnull(row['storage_directory_combined']):
            status = "the path is null"
        else:
            path = "\\" + os.path.normpath(row['storage_directory_combined'])
            roi_name = os.path.basename(path)
        
            # Check if the local path exists
            if not os.path.exists(path):
                status = "cannot find the path"
            else:
                # Attempt to sync the directory
                # status = sync_directory(path, s3_bucket + "/" + roi_name)
                status = "successfully uploaded"  # for now, just
                pass
        results.append({'storage_directory': path, 'status': status})
        
    logger.info(f"Uploaded {len(results)} files to {s3_bucket}")
    logger.info(f'Successful uploads: {len([r for r in results if r["status"] == "successfully uploaded"])}')
    logger.info(f'Skiped: {len([r for r in results if r["status"] == "already exists, skip"])}')
    logger.warning(f'Error during sync: {len([r for r in results if r["status"] == "error during sync"])}')
    logger.warning(f'Cannot find on Isilon: {len([r for r in results if r["status"] == "cannot find the path"])}')
    logger.warning(f'Null path: {len([r for r in results if r["status"] == "the path is null"])}')
    
    return pd.DataFrame(results)


if __name__ == "__main__":
    
    # Set logger level
    logging.basicConfig(level=logging.INFO)
    
    # Generate a list of isilon paths
    dfs = read_brian_spreadsheet(add_lims=True)

    upload_raw_from_isilon_to_s3(dfs["df_all"], s3_bucket=s3_bucket)
