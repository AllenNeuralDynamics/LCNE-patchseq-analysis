"""I/O utilities for eFEL features."""

import logging
import tempfile

import pandas as pd

from LCNE_patchseq_analysis import RESULTS_DIRECTORY
from LCNE_patchseq_analysis.pipeline_util.s3 import S3_PATH_BASE, s3

logger = logging.getLogger(__name__)


def save_dict_to_hdf5(data_dict: dict, filename: str, compress: bool = False):
    """
    Save a dictionary of DataFrames to an HDF5 file using pandas.HDFStore.

    Args:
        data_dict: dict of {str: pd.DataFrame}
        filename: path to .h5 file
        compress: whether to use compression (blosc, level 9)
    """
    with pd.HDFStore(filename, mode="w") as store:
        for key, df in data_dict.items():
            if compress:
                store.put(key, df, format="table", complib="blosc", complevel=9)
            else:
                store.put(key, df)


def load_dict_from_hdf5(filename: str):
    """
    Load a dictionary of DataFrames from an HDF5 file using pandas.HDFStore.

    Args:
        filename: path to .h5 file

    Returns:
        dict: Dictionary of DataFrames
    """
    with pd.HDFStore(filename, mode="r") as store:
        dict_key = [key.replace("/", "") for key in store.keys()]
        return {key: store[key] for key in dict_key}


def load_efel_features_from_roi(roi_id: str, if_from_s3=False):
    """
    Load eFEL features from ROI ID.

    Args:
        roi_id: The ROI ID to load features for
        if_from_s3: If True, load from S3 instead of local file

    Returns:
        Dictionary of DataFrames containing eFEL features
    """
    if if_from_s3:
        s3_path = f"{S3_PATH_BASE}/efel/features/{roi_id}_efel.h5"
        with s3.open(s3_path, "rb") as f:
            with tempfile.NamedTemporaryFile(suffix=".h5") as tmp_file:
                tmp_file.write(f.read())
                tmp_file.flush()
                logger.info(f"Loaded eFEL features from {s3_path} to {tmp_file.name}")
                return load_dict_from_hdf5(tmp_file.name)
    else:
        filename = f"{RESULTS_DIRECTORY}/features/{roi_id}_efel.h5"
        return load_dict_from_hdf5(filename)


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(load_efel_features_from_roi("1212546732", if_from_s3=True).keys())
