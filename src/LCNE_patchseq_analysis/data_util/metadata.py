"""Get session-wise metadata from the JSON files."""

import json
import glob
import pandas as pd

from LCNE_patchseq_analysis import RAW_DIRECTORY

json_name_mapper = {
    "stimulus_summary": "EPHYS_NWB_STIMULUS_SUMMARY",
    "qc": "EPHYS_QC",
    "ephys_fx": "EPHYS_FEATURE_EXTRACTION",
}

def read_json_files(ephys_roi_id="1410790193"):
    """Read json files for the given ephys_roi_id into dicts
    """
    json_dicts = {}
    for key in json_name_mapper:
        json_files = glob.glob(f"{RAW_DIRECTORY}/Ephys_Roi_Result_{ephys_roi_id}/*{json_name_mapper[key]}*output.json")
        if len(json_files) == 0:
            raise FileNotFoundError(f"JSON file not found for {key} in {ephys_roi_id}")
        elif len(json_files) > 1:
            raise ValueError(f"Multiple JSON files found for {key} in {ephys_roi_id}")
        else:
            with open(json_files[0], "r") as f:
                json_dicts[key] = json.load(f)
    return json_dicts


def jsons_to_df(json_dicts):
    """Extract the json dicts to a merged pandas dataframe.
    
    See notes here https://hanhou.notion.site/Output-jsons-1b43ef97e73580f1ae62d3d81039c1a2
    """
    
    df_sweep_features = pd.DataFrame(json_dicts["stimulus_summary"]["sweep_features"])
    df_qc = pd.DataFrame(json_dicts["qc"]["sweep_states"])
    df_ephys_fx = pd.DataFrame(json_dicts["ephys_fx"]["sweep_records"])
    
    df_merged = df_sweep_features.merge(
        df_qc,
        on="sweep_number",
        how="left",
    ).merge(
        df_ephys_fx[["sweep_number", "peak_deflection",	"num_spikes"]],
        on="sweep_number",
        how="left"
    )
    return df_merged

if __name__ == "__main__":
    json_dicts = read_json_files(ephys_roi_id="1410790193")
    pass
    