"""Get metadata"""

import logging
import os

import pandas as pd

from LCNE_patchseq_analysis.pipeline_util.lims import get_lims_LCNE_patchseq

metadata_path = os.path.expanduser(R"~/Downloads/IVSCC_LC_summary.xlsx")
cell_pinning_on_VAST = R"\\allen\programs\celltypes\workgroups\mousecelltypes\cell_pinning\soma_pins.csv"
logger = logging.getLogger(__name__)


def read_brian_spreadsheet(file_path=metadata_path, add_lims=True):
    """Read metadata and ephys features from Brian's spreadsheet

    Assuming IVSCC_LC_summary.xlsx is downloaded at file_path

    Args:
        file_path (str): Path to the metadata spreadsheet
        add_lims (bool): Whether to add LIMS data
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    logger.info(f"Reading metadata from {file_path}...")
    tab_names = pd.ExcelFile(file_path).sheet_names

    # Get the master table
    tab_master = [name for name in tab_names if "master" in name.lower()][0]
    df_tab_master = pd.read_excel(file_path, sheet_name=tab_master)

    # Get ephys features
    tab_ephys_fx = [name for name in tab_names if "ephys_fx" in name.lower()][0]
    df_tab_ephys_fx = pd.read_excel(file_path, sheet_name=tab_ephys_fx)

    # Merge the tables
    df_merged = df_tab_master.merge(
        df_tab_ephys_fx.rename(
            columns={
                "failed_seal": "failed_no_seal",
                "failed_input_access_resistance": "failed_bad_rs",
            }
        ),
        on="cell_specimen_id",
        how="outer",
        suffixes=("_tab_master", "_tab_ephys_fx"),
    ).sort_values("Date", ascending=False)

    if add_lims:
        logger.info("Querying and adding LIMS data...")
        df_lims = get_lims_LCNE_patchseq()
        df_merged = df_merged.merge(
            df_lims.rename(
                columns={
                    "specimen_name": "jem-id_cell_specimen",
                    "specimen_id": "cell_specimen_id",
                }
            ),
            on="jem-id_cell_specimen",
            how="outer",  # Do an outer join to keep all rows
            suffixes=("_tab_master", "_lims"),
            indicator=True,
        )

        df_merged["_merge"] = df_merged["_merge"].replace(
            {"left_only": "spreadsheet_only", "right_only": "lims_only", "both": "both"}
        )
        df_merged.rename(columns={"_merge": "spreadsheet_or_lims"}, inplace=True)

        # Combine storage directories: use LIMS if available, otherwise use master
        df_merged["storage_directory_combined"] = df_merged["storage_directory_lims"].combine_first(
            df_merged["storage_directory_tab_master"]
        )

        logger.info(
            f"Merged LIMS to spreadsheet, total {len(df_merged)} rows: "
            f"{len(df_merged[df_merged['spreadsheet_or_lims'] == 'both'])} in both, "
            f"{len(df_merged[df_merged['spreadsheet_or_lims'] == 'spreadsheet_only'])} "
            f"in spreadsheet only, "
            f"{len(df_merged[df_merged['spreadsheet_or_lims'] == 'lims_only'])} in LIMS only.\n"
        )

    # --- Parse more metadata from the spreadsheet ---
    # Parse experimenter name
    def _map_experimenter(cell_container):
        """Map cell container to experimenter name"""
        if pd.isnull(cell_container):
            return "unknown"
        if cell_container.startswith("P") and len(cell_container) > 1:
            return cell_container[:2]
        return "unknown"
    df_merged["experimenter"] = df_merged["jem-id_patched_cell_container"].map(_map_experimenter)
    
    # Compute age (in days) from date of birth and recording date
    def _compute_age(row):
        """Compute age in days from date of birth and recording date"""
        if pd.isnull(row["date_of_birth"]) or pd.isnull(row["recording_date"]):
            return None
        return (row["recording_date"] - row["date_of_birth"]).days
    df_merged["age_days"] = df_merged.apply(_compute_age, axis=1)
    
    # Merge in CCF coordinates from VAST
    df_pinned_ccf = read_pinned_ccf_from_vast()
    if not df_pinned_ccf.empty:
        df_merged = df_merged.merge(
            df_pinned_ccf,
            left_on="cell_specimen_id_tab_master",
            right_on="cell_specimen_id",
            how="left",
        )

    return {
        "df_merged": df_merged,
        "df_tab_master": df_tab_master,
        "df_tab_ephys_fx": df_tab_ephys_fx,
        **({"df_lims": df_lims} if add_lims else {}),
    }


def read_pinned_ccf_from_vast(file_path=cell_pinning_on_VAST):
    # Read csv from VAST
    if not os.path.exists(file_path):
        logger.warning(f"Pinned coordinates file not found at {file_path}")
        return pd.DataFrame()
    df_pinned = pd.read_csv(file_path)
    logger.info(f"Read {len(df_pinned)} rows from {file_path}...")
    return df_pinned
    

def cross_check_metadata(df, source, check_separately=True):
    """Cross-check metadata between source and master tables

    source in ["tab_ephys_fx", "lims"]

    Args:
        df (pd.DataFrame): The merged dataframe
        source (str): The source table to cross-check with the master table
        check_separately (bool): Whether to check each column separately or all columns together
    """
    source_columns = [
        col for col in df.columns if source in col and col not in ["spreadsheet_or_lims"]
    ]  # Exclude merge indicator column
    master_columns = [col.replace(source, "tab_master") for col in source_columns]

    logger.info("")
    logger.info("-" * 50)
    logger.info(f"Cross-checking metadata between {source} and master tables...")
    logger.info(f"Source columns: {source_columns}")
    logger.info(f"Master columns: {master_columns}")

    # Find out inconsistencies between source and master, if both of them are not null
    if check_separately:
        df_inconsistencies_all = {}
        for source_col, master_col in zip(source_columns, master_columns):
            df_inconsistencies = df.loc[
                (
                    df[source_col].notnull()
                    & df[master_col].notnull()
                    & (df[source_col] != df[master_col])
                ),
                ["Date", "jem-id_cell_specimen", master_col, source_col],
            ]
            if len(df_inconsistencies) > 0:
                logger.warning(
                    f"Found {len(df_inconsistencies)} inconsistencies between "
                    f"{source_col} and {master_col}:"
                )
                logger.warning(df_inconsistencies.to_string(index=False))
                logger.warning("")
            else:
                logger.info(f"All good between {source_col} and {master_col}!")
            df_inconsistencies_all[source_col] = df_inconsistencies
        return df_inconsistencies_all
    else:
        df_inconsistencies = df.loc[
            (
                df[source_columns].notnull()
                & df[source_columns].notnull()
                & (df[source_columns].to_numpy() != df[master_columns].to_numpy())
            ).any(axis=1),
            ["Date", "jem-id_cell_specimen"] + master_columns + source_columns,
        ]
        if len(df_inconsistencies) > 0:
            logger.warning(
                f"Found {len(df_inconsistencies)} inconsistencies between "
                f"{source} and master tables:"
            )
            logger.warning(df_inconsistencies.to_string(index=False))
            logger.warning("")
        else:
            logger.info(f"All good between {source} and master tables!")
        return df_inconsistencies


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    dfs = read_brian_spreadsheet()

    for source in ["tab_ephys_fx", "lims"]:
        df_inconsistencies = cross_check_metadata(dfs["df_merged"], source, check_separately=True)
