"""Get metadata"""

import logging
import os

import pandas as pd

from LCNE_patchseq_analysis.pipeline_util.lims import get_lims_LCNE_patchseq

metadata_path = os.path.expanduser(R"~\Downloads\IVSCC_LC_summary.xlsx")
logger = logging.getLogger(__name__)


def read_brian_spreadsheet(file_path=metadata_path, add_lims=True):
    """Read metadata, cell xyz coordinates, and ephys features from Brian's spreadsheet

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
    tab_master = [name for name in tab_names if "updated" in name.lower()][0]
    df_tab_master = pd.read_excel(file_path, sheet_name=tab_master)

    # Get xyz coordinates
    tab_xyz = [name for name in tab_names if "xyz" in name.lower()][0]
    df_tab_xyz = pd.read_excel(file_path, sheet_name=tab_xyz)

    # Get ephys features
    tab_ephys_fx = [name for name in tab_names if "ephys_fx" in name.lower()][0]
    df_tab_ephys_fx = pd.read_excel(file_path, sheet_name=tab_ephys_fx)

    # Merge the tables
    df_merged = (
        df_tab_master.merge(
            df_tab_xyz.rename(
                columns={
                    "specimen_name": "jem-id_cell_specimen",
                    "structure_acronym": "Annotated structure",
                }
            ),
            on="jem-id_cell_specimen",
            how="outer",
            suffixes=("_tab_master", "_tab_xyz"),
        )
        .merge(
            df_tab_ephys_fx.rename(
                columns={
                    "failed_seal": "failed_no_seal",
                    "failed_input_access_resistance": "failed_bad_rs",
                }
            ),
            on="cell_specimen_id",
            how="outer",
            suffixes=("_tab_master", "_tab_ephys_fx"),
        )
        .sort_values("Date", ascending=False)
    )

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

    return {
        "df_merged": df_merged,
        "df_tab_master": df_tab_master,
        "df_tab_xyz": df_tab_xyz,
        "df_tab_ephys_fx": df_tab_ephys_fx,
        **({"df_lims": df_lims} if add_lims else {}),
    }


def cross_check_metadata(df, source, check_separately=True):
    """Cross-check metadata between source and master tables

    source in ["tab_xyz", "tab_ephys_fx", "lims"]

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

    for source in ["tab_xyz", "tab_ephys_fx", "lims"]:
        df_inconsistencies = cross_check_metadata(dfs["df_merged"], source, check_separately=True)
