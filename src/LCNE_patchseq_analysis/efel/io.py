"""I/O utilities for eFEL features."""

import pandas as pd

from LCNE_patchseq_analysis import RESULTS_DIRECTORY


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


def load_efel_features_from_roi(roi_id: str):
    """Load eFEL features from ROI ID."""
    filename = f"{RESULTS_DIRECTORY}/features/{roi_id}_efel.h5"
    return load_dict_from_hdf5(filename)
