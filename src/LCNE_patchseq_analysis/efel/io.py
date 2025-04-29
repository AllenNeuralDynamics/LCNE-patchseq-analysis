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


# Note: load_efel_features_from_roi function has been moved to pipeline_util/s3.py


if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)
    print(load_efel_features_from_roi("1212546732", if_from_s3=True).keys())
