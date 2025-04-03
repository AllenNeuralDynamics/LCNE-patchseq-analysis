"""Get session-wise metadata from the JSON files."""

import glob
import json
import logging

import pandas as pd

from LCNE_patchseq_analysis import RAW_DIRECTORY

logger = logging.getLogger(__name__)

json_name_mapper = {
    "stimulus_summary": "EPHYS_NWB_STIMULUS_SUMMARY",
    "qc": "EPHYS_QC",
    "ephys_fx": "EPHYS_FEATURE_EXTRACTION",
}


def read_json_files(ephys_roi_id="1410790193"):
    """Read json files for the given ephys_roi_id into dicts"""
    json_dicts = {}
    for key in json_name_mapper:
        json_files = glob.glob(
            f"{RAW_DIRECTORY}/Ephys_Roi_Result_{ephys_roi_id}/*{json_name_mapper[key]}*output.json"
        )
        if len(json_files) == 0:
            if key == "ephys_fx":
                logger.warning(
                    f"ephys_fx json file not found for {key} in {ephys_roi_id}, skipping.."
                )
                continue
            raise FileNotFoundError(f"JSON file not found for {key} in {ephys_roi_id}")
        elif len(json_files) > 1:
            logger.warning(
                f"Multiple JSON files found for {key} in {ephys_roi_id}, using the first one"
            )

        with open(json_files[0], "r") as f:
            json_dicts[key] = json.load(f)
        logger.info(f"Loaded {key} from {json_files[0]}")
    return json_dicts


def jsons_to_df(json_dicts):
    """Extract the json dicts to a merged pandas dataframe.

    See notes here https://hanhou.notion.site/Output-jsons-1b43ef97e73580f1ae62d3d81039c1a2
    """

    df_sweep_features = pd.DataFrame(json_dicts["stimulus_summary"]["sweep_features"])

    # If is empty, return None
    if len(df_sweep_features) == 0:
        return None

    df_qc = pd.DataFrame(json_dicts["qc"]["sweep_states"])

    if "ephys_fx" not in json_dicts:
        df_ephys_fx = pd.DataFrame(
            {
                "sweep_number": df_sweep_features["sweep_number"],
                "peak_deflection": [None] * len(df_sweep_features),
                "num_spikes": [None] * len(df_sweep_features),
            }
        )
    else:
        df_ephys_fx = pd.DataFrame(json_dicts["ephys_fx"]["sweep_records"])

    df_merged = df_sweep_features.merge(
        df_qc,
        on="sweep_number",
        how="left",
    ).merge(
        df_ephys_fx[["sweep_number", "peak_deflection", "num_spikes"]],
        on="sweep_number",
        how="left",
    )
    logger.info(f"Merged sweep metadata, shape: {df_merged.shape}")
    return df_merged


def load_ephys_metadata():
    """Load ephys metadata

    Per discussion with Brian, we should only look at those in the spreadsheet.
    https://www.notion.so/hanhou/LCNE-patch-seq-analysis-1ae3ef97e735808eb12ec452d2dc4369?pvs=4#1ba3ef97e73580ac9a5ee6e53e9b3dbe  # noqa: E501
    """
    df = pd.read_csv(RAW_DIRECTORY + "/df_metadata_merged.csv")
    df = df.query("spreadsheet_or_lims in ('both', 'spreadsheet_only')").copy()

    # Rename "Crus 1" to "Crus1"
    df.loc[
        df["injection region"].astype(str).str.contains("Crus", na=False),
        "injection region",
    ] = "Crus 1"

    # Change columns with roi_id to str(int())
    for col in ["ephys_roi_id_tab_master", "ephys_roi_id_lims"]:
        df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) else "")
    return df


if __name__ == "__main__":
    json_dicts = read_json_files(
        # ephys_roi_id="1410790193"  # Examle cell that has ephys_fx
        ephys_roi_id="1417382638",  # Example cell that does not have ephys_fx
    )
    df_merged = jsons_to_df(json_dicts)
    print(df_merged.head())

    df_meta = load_ephys_metadata()
    print(df_meta.head())
