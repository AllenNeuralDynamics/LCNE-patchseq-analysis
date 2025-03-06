"""Get metadata"""

import os

import pandas as pd

from LCNE_patchseq_analysis.data_util.lims import get_lims_LCNE_patchseq

metadata_path = os.path.expanduser("~\Downloads\IVSCC_LC_summary.xlsx")


def read_brian_spreadsheet(file_path=metadata_path, add_lims=True):
    """Read metadata, cell xyz coordinates, and ephys features from Brian's spreadsheet

    Assuming IVSCC_LC_summary.xlsx is downloaded at file_path

    Args:
        file_path (str): Path to the metadata spreadsheet
        add_lims (bool): Whether to add LIMS data
    """

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found at {file_path}")

    tab_names = pd.ExcelFile(file_path).sheet_names

    # Get the master table
    tab_master = [name for name in tab_names if "updated" in name.lower()][0]
    df_master = pd.read_excel(file_path, sheet_name=tab_master)

    # Get xyz coordinates
    tab_xyz = [name for name in tab_names if "xyz" in name.lower()][0]
    df_xyz = pd.read_excel(file_path, sheet_name=tab_xyz)

    # Get ephys features
    tab_ephys_fx = [name for name in tab_names if "ephys_fx" in name.lower()][0]
    df_ephys_fx = pd.read_excel(file_path, sheet_name=tab_ephys_fx)

    # Merge the tables
    df_all = (
        df_master.merge(
            df_xyz.rename(
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
            df_ephys_fx.rename(
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
        df_lims = get_lims_LCNE_patchseq()
        df_all = df_all.merge(
            df_lims,
            left_on="jem-id_cell_specimen",
            right_on="specimen_name",
            how="left",
            suffixes=("_master", "_lims"),
        )

    return {
        "df_all": df_all,
        "df_master": df_master,
        "df_xyz": df_xyz,
        "df_ephys_fx": df_ephys_fx,
        **({"df_lims": df_lims} if add_lims else {}),
    }


if __name__ == "__main__":
    dfs = read_brian_spreadsheet()
    print(dfs["df_all"].head())
