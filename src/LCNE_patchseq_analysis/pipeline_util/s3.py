"""
Utility functions for S3.
"""

import requests
import s3fs

s3 = s3fs.S3FileSystem(anon=True)

S3_PUBLIC_URL_BASE = "https://aind-scratch-data.s3.us-west-2.amazonaws.com/aind-patchseq-data"

S3_PATH_BASE = "aind-scratch-data/aind-patchseq-data/"


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


if __name__ == "__main__":
    print(get_public_url_sweep("1212546732", 46))
