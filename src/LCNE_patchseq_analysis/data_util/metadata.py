"""Get session-wise metadata from the JSON files."""

import glob
import json
import logging

import pandas as pd

from LCNE_patchseq_analysis import RAW_DIRECTORY
from LCNE_patchseq_analysis.pipeline_util.s3 import (
    get_public_efel_cell_level_stats,
    get_public_seq_preselected,
    get_public_mapmycells,
)

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


def load_ephys_metadata(if_from_s3=False, if_with_seq=False, combine_roi_ids=False):
    """Load ephys metadata

    Per discussion with Brian, we should only look at those in the spreadsheet.
    https://www.notion.so/hanhou/LCNE-patch-seq-analysis-1ae3ef97e735808eb12ec452d2dc4369?pvs=4#1ba3ef97e73580ac9a5ee6e53e9b3dbe  # noqa: E501

    Args:
        if_from_s3: If True, load the cell level stats from eFEL output
                            (Brian's spreadsheet + eFEL stats).
                      else, load the downloaded Brian's spreadsheet only.
        if_with_seq: If True, merge in sequencing data from seq_preselected.csv,
                     matching on the exp_component_name column.
        combine_roi_ids: If True, combine "ephys_roi_id_lims" into "ephys_roi_id_tab_master".
    """
    # -- Load the cell level stats from eFEL output --
    if if_from_s3:
        df = get_public_efel_cell_level_stats()

        # -- Convert ephys_roi_id to str(int()) --
        df["ephys_roi_id"] = df["ephys_roi_id"].apply(
            lambda x: str(int(x)) if pd.notnull(x) else ""
        )
        
        # -- Parse mouse line --
        # In "jem-id_cell_specimen" field, extract the string before the first ;
        # this is the mouse line
        df["mouse_line"] = df["jem-id_cell_specimen"].str.split(";").str[0]
        df["mouse_line"] = df["mouse_line"].apply(
            lambda x: "C57BL6J" if isinstance(x, str) and "C57BL6J" in x else x
        )

        # Merge sequencing data if requested
        if if_with_seq:
            try:
                logger.info("Loading sequencing data from S3...")
                # -- Merge sequencing data from S3 --
                df_seq = get_public_seq_preselected()
                
                # Add "gene_" columns names in df_seq to the dataframe
                df_seq = df_seq.rename(
                    columns=lambda x: f"gene_{x} (log_normed)" if x != "cell_specimen_id" else x
                )

                # Perform the merge on cell_specimen_id
                df = df.merge(
                    df_seq,
                    left_on="cell_specimen_id",
                    right_on="cell_specimen_id",
                    how="left",
                )

                # Log the merge results
                merged_count = df["cell_specimen_id"].notna().sum()
                logger.info(
                    f"Successfully merged sequencing data for {merged_count} out of {len(df)} cells"
                )
                
                # -- Merge MapMyCells results from S3 --
                df_mapmycells = get_public_mapmycells()
                
                # Add "mapmycells_" prefix to the columns names in df_mapmycells
                df_mapmycells = df_mapmycells.rename(
                    columns=lambda x: f"mapmycells_{x}"
                )
                
                # Merge MapMyCells data into df
                df = df.merge(
                    df_mapmycells, left_on="exp_component_name",
                    right_on="mapmycells_cell_id",
                    how="left",
                ).drop(columns=["mapmycells_cell_id"])
                
                # Set nan's in "subclass_category" to "seq_data_not_available"
                df["mapmycells_subclass_category"] = df["mapmycells_subclass_category"
                                                        ].fillna("seq_data_not_available")
                
            except FileNotFoundError as e:
                logger.warning(f"Could not load sequencing data: {e}")
            except Exception as e:
                logger.error(f"Error merging sequencing data: {e}")

        return df

    # -- Load the downloaded Brian's spreadsheet only --
    df = pd.read_csv(RAW_DIRECTORY + "/df_metadata_merged.csv")
    df = df.query("spreadsheet_or_lims in ('both', 'spreadsheet_only')").copy()

    # Format injection region
    df.loc[
        df["injection region"].astype(str).str.contains("Crus", na=False),
        "injection region",
    ] = "Crus 1"
    df["injection region"] = df["injection region"].apply(format_injection_region)

    # Convert width columns to ms
    df.loc[:, df.columns.str.contains("width")] = df.loc[:, df.columns.str.contains("width")] * 1000

    # Combine roi_ids (when ephys_roi_id already exists on LIMS but not updated on spreadsheet)
    if combine_roi_ids:
        # Combine "ephys_roi_id_lims" into "ephys_roi_id_tab_master"
        df["ephys_roi_id_tab_master"] = df["ephys_roi_id_tab_master"].combine_first(
            df["ephys_roi_id_lims"]
        )

    # --- Temporary fix @ 2025-04-09 ---
    # The xyz are removed from the spreadsheet and for now I still don't know how to get from LIMS
    # So I'm merging [x_tab_master, y_tab_master, z_tab_master] from the
    # df_metadata_merged_20250409.csv
    df_temp = pd.read_csv(RAW_DIRECTORY + "/df_metadata_merged_20250409.csv").copy()
    df = df.merge(
        df_temp[["ephys_roi_id_tab_master", "x_tab_master", "y_tab_master", "z_tab_master"]],
        on="ephys_roi_id_tab_master",
        how="left",
    )

    # Fix missing LC_targeting (set to "retro" if "injection region" is not "Non-Retro")
    df.loc[df["injection region"] != "Non-Retro", "LC_targeting"] = "retro"

    # Change columns with roi_id to str(int())
    for col in ["ephys_roi_id_tab_master", "ephys_roi_id_lims"]:
        df[col] = df[col].apply(lambda x: str(int(x)) if pd.notnull(x) else "")

    return df


def format_injection_region(x):
    if x != x:
        return "Non-Retro"
    if "pl" in x.lower():
        return "Cortex"
    return x


if __name__ == "__main__":
    json_dicts = read_json_files(
        # ephys_roi_id="1410790193"  # Examle cell that has ephys_fx
        ephys_roi_id="1417382638",  # Example cell that does not have ephys_fx
    )
    df_merged = jsons_to_df(json_dicts)
    print(df_merged.head())

    df_meta = load_ephys_metadata()
    print(df_meta.head())
