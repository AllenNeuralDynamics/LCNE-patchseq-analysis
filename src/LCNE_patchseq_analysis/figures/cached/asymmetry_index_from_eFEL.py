import os
import re

import numpy as np
import pandas as pd

from LCNE_patchseq_analysis import RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures import GLOBAL_FILTER, set_plot_style
from LCNE_patchseq_analysis.figures.cached.fig_3c import (
    _generate_multi_feature_scatter_plots,
)
from LCNE_patchseq_analysis.figures.util import save_figure
from LCNE_patchseq_analysis.population_analysis.anova import anova_features


def add_efel_asymmetry_columns(df_meta: pd.DataFrame) -> None:
    """Add eFEL rise/fall asymmetry columns in-place."""
    # Find rise/fall pairs for each eFEL metric and compute asymmetry.
    rise_cols = {}
    fall_cols = {}

    for col in df_meta.columns:
        if not isinstance(col, str):
            continue
        if not col.startswith("efel_") or " @ " not in col:
            continue
        metric, suffix = col.split(" @ ", 1)
        match = re.match(r"^(efel_.+?)_(rise|fall)(.*)$", metric)
        if not match:
            continue
        base = f"{match.group(1)}{match.group(3)}"
        key = f"{base} @ {suffix}"
        if match.group(2) == "rise":
            rise_cols[key] = col
        else:
            fall_cols[key] = col

    for key in sorted(set(rise_cols) & set(fall_cols)):
        base, suffix = key.split(" @ ", 1)
        new_col = f"{base}_asymmetry @ {suffix}"
        rise_vals = pd.to_numeric(df_meta[rise_cols[key]], errors="coerce")
        fall_vals = pd.to_numeric(df_meta[fall_cols[key]], errors="coerce")
        df_meta[new_col] = np.where(fall_vals != 0, rise_vals / fall_vals, np.nan)


def format_asymmetry_name(col_name: str) -> str:
    """Convert raw column names into readable plot labels."""
    # Convert raw column names into readable plot labels.
    display = col_name.replace("efel_", "")
    display = display.replace("AP_asymmetry_index", "asymmetry_index")
    display = display.replace("first_spike_", "first_spike ")
    display = display.replace("second_spike_", "second_spike ")
    display = display.replace("last_spike_", "last_spike ")
    if not display.startswith(("first_spike ", "second_spike ", "last_spike ")):
        display = f"first_spike {display}"
    return display


def main(if_save_figure: bool = True):
    """Run the eFEL asymmetry analysis and plot the summary figure."""
    # 1) Load metadata and apply the global filter used by other figures.
    set_plot_style(base_size=12, font_family="Helvetica")

    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    df_meta_filtered = df_meta.query(GLOBAL_FILTER).copy()

    # 2) Compute asymmetry columns from all available rise/fall eFEL pairs.
    add_efel_asymmetry_columns(df_meta_filtered)
    asymmetry_cols = sorted([col for col in df_meta_filtered.columns if "_asymmetry @" in col])
    asymmetry_features = [{col: format_asymmetry_name(col)} for col in asymmetry_cols]

    # 3) Run ANCOVA to test projection effects while controlling for y.
    df_anova = anova_features(
        df_meta_filtered,
        features=asymmetry_cols,
        cat_col="injection region",
        cont_col="y",
        adjust_p=True,
        anova_typ=2,
    )

    # 4) Generate scatter plots and optionally save them.
    fig, axes = _generate_multi_feature_scatter_plots(
        df_meta=df_meta_filtered,
        features=asymmetry_features,
        df_anova=df_anova,
        filename="sup_fig_asymmetry_index",
        if_save_figure=False,
        n_cols=4,
    )
    if if_save_figure:
        output_dir = os.path.join(RESULTS_DIRECTORY, "figures", "asymmetry_eFEL_summary")
        save_figure(
            fig,
            output_dir=output_dir,
            filename="sup_fig_asymmetry_index",
            dpi=300,
            formats=("png", "svg"),
        )
    return fig, axes, df_anova


if __name__ == "__main__":
    main(if_save_figure=True)
