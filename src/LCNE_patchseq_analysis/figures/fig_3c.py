import matplotlib.pyplot as plt
import numpy as np
from numpy.__config__ import show
import pandas as pd
import seaborn as sns
from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata

from LCNE_patchseq_analysis.figures.util import save_figure, generate_violin_plot, generate_scatter_plot
from LCNE_patchseq_analysis.figures import sort_region


def figure_3c_tau_comparison(
    df_meta: pd.DataFrame, filter_query: str | None = None, if_save_figure: bool = True, ax=None
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
        df_meta = df_meta.query(filter_query).copy()

    df_meta["ipfx_tau"] = df_meta["ipfx_tau"] * 1000  # convert to ms
    # fig, ax = generate_violin_plot(
    #     df_to_use=df_meta,
    #     y_col="ipfx_tau",
    #     color_col="injection region",
    #     color_palette_dict=REGION_COLOR_MAPPER,
    #     ax=ax
    # )
    fig, ax = generate_scatter_plot(
        df=df_meta,
        y_col="ipfx_tau",
        x_col="y",
        color_col="injection region",
        color_palette=REGION_COLOR_MAPPER,
        plot_linear_regression=True,
        show_marginal=True,
        marginal_kind="kde",
        ax=ax
    )

    ax.set_xlabel("Dorsal-ventral (μm)")
    ax.set_ylabel("Time constant (ms)")

    if if_save_figure:
        save_figure(fig, filename="fig_3c_violinplot_ipfx_tau", dpi=300, formats=("png", "pdf"))
        print("Figure saved as fig_3c_violinplot_ipfx_tau.png/.pdf")
    return fig, ax


def figure_3c_latency_comparison(
    df_meta: pd.DataFrame, filter_query: str | None = None, if_save_figure: bool = True, ax=None
):
    """
    Generate and save violin plot for ipfx_latency grouped by injection region (Figure 3B).
    Args:
        df_meta: DataFrame containing metadata.
        filter_query: Optional pandas query string to filter the metadata.
    """
    if filter_query:
        df_meta = df_meta.query(filter_query).copy()

    # fig, ax = generate_violin_plot(
    #     df_to_use=df_meta,
    #     y_col="ipfx_latency_rheo",
    #     color_col="injection region",
    #     color_palette_dict=REGION_COLOR_MAPPER,
    #     ax=ax
    # )
    fig, ax = generate_scatter_plot(
        df=df_meta,
        y_col="ipfx_latency_rheo",
        x_col="y",
        color_col="injection region",
        color_palette=REGION_COLOR_MAPPER,
        plot_linear_regression=True,
        show_marginal=True,
        marginal_kind="kde",
        ax=ax
    )

    ax.set_xlabel("Dorsal-ventral (μm)")
    ax.set_ylabel("Latency to first spike\nat rheobase (s)")

    if if_save_figure:
        save_figure(fig, filename="fig_3c_violinplot_ipfx_latency", dpi=300, formats=("png", "pdf"))
        print("Figure saved as fig_3c_violinplot_ipfx_latency.png/.pdf")
    return fig, ax


if __name__ == "__main__":
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    from LCNE_patchseq_analysis.figures import GLOBAL_FILTER
    figure_3c_tau_comparison(df_meta, GLOBAL_FILTER)
    figure_3c_latency_comparison(df_meta, GLOBAL_FILTER)
