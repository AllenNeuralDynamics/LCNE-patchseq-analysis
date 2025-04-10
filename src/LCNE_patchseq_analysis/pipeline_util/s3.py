"""
Utility functions for S3.
"""

import logging
import os
import subprocess

import pandas as pd
import requests
import s3fs

logger = logging.getLogger(__name__)

s3 = s3fs.S3FileSystem(anon=True)  # All on public bucket

S3_PUBLIC_URL_BASE = "https://aind-scratch-data.s3.us-west-2.amazonaws.com/aind-patchseq-data"
S3_PATH_BASE = "aind-scratch-data/aind-patchseq-data"


def sync_directory(local_dir, destination, if_copy=False):
    """
    Sync the local directory with the given S3 destination using aws s3 sync.
    Returns a status string based on the command output.
    """
    try:
        if if_copy:
            # Run aws s3 cp command and capture the output
            result = subprocess.run(
                ["aws", "s3", "cp", local_dir, destination], capture_output=True, text=True
            )
        else:
            # Run aws s3 sync command and capture the output
            result = subprocess.run(
                ["aws", "s3", "sync", local_dir, destination], capture_output=True, text=True
            )
        output = result.stdout + result.stderr

        # Check output: if "upload:" appears, files were sent;
        # otherwise, assume that nothing needed uploading.
        if "upload:" in output:
            logger.info(f"Uploaded {local_dir} to {destination}!")
            return "successfully uploaded"
        else:
            logger.info(output)
            logger.info(f"Already exists, skip {local_dir}.")
            return "already exists, skip"
    except Exception as e:
        return f"error during sync: {e}"


def check_s3_public_url_exists(s3_url: str) -> bool:
    """Check if a given s3 url exists."""
    response = requests.get(s3_url)
    return response.status_code == 200


def get_public_url_sweep(ephys_roi_id: str, sweep_number: int) -> str:
    """Get the public URL for a sweep."""

    s3_sweep = (
        f"{S3_PUBLIC_URL_BASE}/efel/plots/{ephys_roi_id}/{ephys_roi_id}_sweep_{sweep_number}.png"
    )
    s3_spikes = (
        f"{S3_PUBLIC_URL_BASE}/efel/plots/{ephys_roi_id}/"
        f"{ephys_roi_id}_sweep_{sweep_number}_spikes.png"
    )

    # Check if the file exists on s3 public
    urls = {}
    if check_s3_public_url_exists(s3_sweep):
        urls["sweep"] = s3_sweep
    if check_s3_public_url_exists(s3_spikes):
        urls["spikes"] = s3_spikes
    return urls


def get_public_efel_cell_level_stats():
    """Get the cell level stats that are generated by eFEL (and merged with the metadata)."""
    csv_url = f"{S3_PUBLIC_URL_BASE}/efel/cell_stats/cell_level_stats.csv"
    if check_s3_public_url_exists(csv_url):
        return pd.read_csv(csv_url)
    else:
        raise FileNotFoundError(f"Cell level stats CSV file not found at {csv_url}")


def get_public_url_cell_summary(ephys_roi_id: str, if_check_exists: bool = True) -> str:
    """Get the public URL for a cell summary plot."""
    s3_url = f"{S3_PUBLIC_URL_BASE}/efel/cell_stats/{ephys_roi_id}_cell_summary.png"
    if if_check_exists:
        if check_s3_public_url_exists(s3_url):
            return s3_url
        else:
            return None
    else:
        return s3_url
    
    
def get_public_representative_spikes() -> pd.DataFrame:
    """Get the representative spikes for a cell."""
    s3_url = f"{S3_PUBLIC_URL_BASE}/efel/cell_stats/cell_level_spike_waveforms.pkl"
    if check_s3_public_url_exists(s3_url):
        return pd.read_pickle(s3_url)
    else:
        raise FileNotFoundError(f"Pickle file not found at {s3_url}")


if __name__ == "__main__":
    # print(get_public_url_sweep("1212546732", 46))
    print(get_public_efel_cell_level_stats())
