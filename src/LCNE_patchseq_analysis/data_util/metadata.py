"""Get metadata
"""

import pandas as pd
import os

file_path = os.path.expanduser("~/capsule/data/IVSCC_LC_summary.xlsx")


def read_brian_spreadsheet(file_path=file_path):
    """ Read metadata, cell xyz coordinates, and ephys features from Brian's spreadsheet
    
    Assuming IVSCC_LC_summary.xlsx is downloaded at file_path
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
            suffixes=("_master", "_xyz"),
        )
        .merge(df_ephys_fx.rename(
                columns={
                    "failed_seal": "failed_no_seal",
                    "failed_input_access_resistance": "failed_bad_rs",
                }
            ), on="cell_specimen_id", how="outer", suffixes=("_master", "_ephys_fx"))
        .sort_values("Date", ascending=False)
    )

    return df_all, df_master, df_xyz, df_ephys_fx

if __name__ == "__main__":
    df_all, df_master, df_xyz, df_ephys_fx = read_brian_spreadsheet()
    print(df_all.head())
    print(df_master.head())
    print(df_xyz.head())
    print(df_ephys_fx.head())
