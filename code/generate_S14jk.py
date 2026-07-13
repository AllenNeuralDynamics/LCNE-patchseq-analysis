"""Reproduce supplementary figure S14 panels j and k.

Panel k (PC1 and membrane time constant cumulative distributions) is rebuilt
directly from ``AIBS_spreadsheet_pub.csv`` with zero dependency on S3 or the
attached data asset: the columns ``spike_waveform_PC1`` and
``membrane_time_constant_ms`` are already precomputed in the CSV.

Panel j (raw example traces) is left as-is and still read from its own data
source; it will be regenerated from the DANDI-hosted NWBs later.
"""

import logging

import pandas as pd

from LCNE_patchseq_analysis.figures.main_pca_tau import figure_s14_jk

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

CSV_PATH = "/data/AIBS_spreadsheet_pub.csv"


def main() -> None:
    logger.info("Loading S14 metadata from %s", CSV_PATH)
    df = pd.read_csv(CSV_PATH)

    # Build the minimal per-cell projection table that panel k expects.
    # `projection_target` doubles as `injection region` because each group label
    # ("Spinal cord" / "Cortex" / "Cerebellum") is itself a member of the
    # corresponding region set used for grouping.
    df_v_proj = pd.DataFrame(
        {
            "ephys_roi_id": df["ephys_roi_id"].astype(str),
            "PCA1": df["spike_waveform_PC1"],
            "membrane_time_constant_ms": df["membrane_time_constant_ms"],
            "injection region": df["projection_target"],
            "Donor": df["Donor"],
        }
    )

    logger.info("Generating figure S14 panels j (example traces) and k (cumulative distributions)...")
    figure_s14_jk(df_v_proj=df_v_proj)


if __name__ == "__main__":
    main()


