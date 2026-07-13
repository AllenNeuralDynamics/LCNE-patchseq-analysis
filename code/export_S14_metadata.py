"""Export the publication metadata for supplementary figure S14 (panels j & k).

Surfaces every cell that goes into S14 together with the columns needed to
reproduce the figure, and writes a single tidy CSV into ``results/``.

Cells:
  * Panel k (CDFs): all cells passing ``GLOBAL_FILTER`` that fall into one of the
    three projection groups (Spinal cord / Cortex / Cerebellum).
  * Panel j (example traces): the three example cells in ``EXAMPLE_CELLS`` (flagged
    with ``example_trace_panel_j``).

Columns needed to reproduce S14:
  * ephys_roi_id            - cell identifier
  * cell_specimen_id        - cell specimen id
  * jem-id_cell_specimen    - cell specimen name
  * Date                    - experiment date
  * Donor                   - mouse identifier (for n-mice counts)
  * projection_target       - grouped target (Spinal cord / Cortex / Cerebellum)
  * slicing plane           - slicing plane
  * spike_waveform_PC1      - first PC of the normalized spike waveform (panel k, left)
  * membrane_time_constant_ms - ipfx_tau in ms (panel k, right)
  * example_trace_panel_j   - True for the panel-j example cells
"""

import logging
import os

import pandas as pd

from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures import GLOBAL_FILTER
from LCNE_patchseq_analysis.figures.example_traces import EXAMPLE_CELLS
from LCNE_patchseq_analysis.figures.main_pca_tau import (
    CB_REGIONS,
    CORTEX_REGIONS,
    SPINAL_REGIONS,
    spike_pca_analysis,
)
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

PROJECTION_GROUPS = [
    ("Spinal cord", SPINAL_REGIONS),
    ("Cortex", CORTEX_REGIONS),
    ("Cerebellum", CB_REGIONS),
]


def _projection_target(injection_region: str) -> str | None:
    for label, region_set in PROJECTION_GROUPS:
        if injection_region in region_set:
            return label
    return None


def main() -> None:
    logger.info("Loading metadata...")
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    df_meta = df_meta.query(GLOBAL_FILTER)
    logger.info("Loaded metadata with shape: %s", df_meta.shape)

    logger.info("Loading spike waveforms...")
    df_spikes = get_public_representative_spikes("average")

    logger.info("Running spike-waveform PCA (panel k analysis)...")
    results = spike_pca_analysis(
        df_meta=df_meta, df_spikes=df_spikes, filtered_df_meta=df_meta
    )
    df_v_proj = results["df_v_proj"]
    tau_col = results["tau_col"]  # "membrane_time_constant_ms"

    # Map each cell to its projection target and keep only the S14 groups.
    df_v_proj = df_v_proj.copy()
    df_v_proj["projection_target"] = df_v_proj["injection region"].map(
        _projection_target
    )
    df_used = df_v_proj[df_v_proj["projection_target"].notna()].copy()

    # Flag the panel-j example cells.
    example_ids = {str(c["ephys_roi_id"]) for c in EXAMPLE_CELLS}
    df_used["ephys_roi_id"] = df_used["ephys_roi_id"].astype(str)
    df_used["example_trace_panel_j"] = df_used["ephys_roi_id"].isin(example_ids)

    # Merge in additional metadata columns from df_meta, keyed by ephys_roi_id.
    extra_cols = [
        "Date",
        "jem-id_cell_specimen",
        "cell_specimen_id",
        "slicing plane",
    ]
    extra_cols = [c for c in extra_cols if c in df_meta.columns]
    extra = df_meta[["ephys_roi_id"] + extra_cols].copy()
    extra["ephys_roi_id"] = extra["ephys_roi_id"].astype(str)
    df_used = df_used.merge(extra, on="ephys_roi_id", how="left")

    # Logical column order: identifiers -> sample metadata -> location ->
    # figure quantities -> panel flag.
    output_cols = [
        "ephys_roi_id",
        "cell_specimen_id",
        "jem-id_cell_specimen",
        "Date",
        "Donor",
        "projection_target",
        "slicing plane",
        "PCA1",
        tau_col,
        "example_trace_panel_j",
    ]
    output_cols = [c for c in output_cols if c in df_used.columns]
    df_out = df_used[output_cols].rename(
        columns={
            "PCA1": "spike_waveform_PC1",
        }
    )

    # Sort for a stable, readable output.
    df_out = df_out.sort_values(
        ["projection_target", "ephys_roi_id"]
    ).reset_index(drop=True)

    output_dir = "/results" if os.getenv("CO_CAPSULE_ID") else os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "results"
    )
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, "AIBS_spreadsheet_pub.csv")
    df_out.to_csv(out_path, index=False)

    logger.info("Saved S14 metadata for %d cells to %s", len(df_out), out_path)
    logger.info(
        "Cells per projection target:\n%s",
        df_out["projection_target"].value_counts().to_string(),
    )
    logger.info(
        "Panel-j example cells found: %d / %d",
        int(df_out["example_trace_panel_j"].sum()),
        len(EXAMPLE_CELLS),
    )


if __name__ == "__main__":
    main()
