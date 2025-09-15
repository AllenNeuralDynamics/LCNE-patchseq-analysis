import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata

from LCNE_patchseq_analysis.figures.util import save_figure, create_violin_plot_matplotlib
from LCNE_patchseq_analysis.figures import sort_region


def figure_3c_tau_comparison(
    df_meta: pd.DataFrame, filter_query: str | None = None, if_save_figure: bool = True
):
    """
    Generate and save violin plot for ipfx_tau grouped by injection region (Figure 3B).
    Args:
        df_meta: DataFrame containing metadata.
        filter_query: Optional pandas query string to filter the metadata.
    Returns:
        (fig, ax): Matplotlib figure and axes, or (None, None) if columns missing.
    """

    # Apply filter if provided
    if filter_query:
        df_meta = df_meta.query(filter_query)

    fig, ax = create_violin_plot_matplotlib(
        df_to_use=df_meta,
        y_col="ipfx_tau",
        color_col="injection region",
        color_palette_dict=REGION_COLOR_MAPPER,
        font_size=12
    )
    if if_save_figure:
        save_figure(fig, filename="fig_3c_violinplot_ipfx_tau", dpi=300, formats=("png", "pdf"))
        print("Figure saved as fig_3c_violinplot_ipfx_tau.png/.pdf")
    return fig, ax


if __name__ == "__main__":
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    from LCNE_patchseq_analysis.figures import GLOBAL_FILTER
    figure_3c_tau_comparison(df_meta, GLOBAL_FILTER)
