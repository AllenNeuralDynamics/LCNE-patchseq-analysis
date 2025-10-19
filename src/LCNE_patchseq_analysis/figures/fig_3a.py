import logging

import pandas as pd

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures.util import generate_ccf_plot, generate_violin_plot, save_figure

# Configure logging
logger = logging.getLogger()


def figure_3a_ccf_sagittal(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
    ax=None,
) -> tuple:
    """Deprecated wrapper around generate_ccf_plot with optional filter and angle.

    Args:
        filter_query: pandas query string to filter the metadata. If None, uses default.
        slicing_angle: 'sagittal' or 'coronal'. Defaults to 'sagittal'.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """

    fig, ax = generate_ccf_plot(
        df_meta, filter_query, view="sagittal", ax=ax, show_marginal_x=True, show_marginal_y=True
    )

    if if_save_figure:
        save_figure(
            fig=fig,
            filename="fig_3a_ccf_sagittal_by_projection",
            formats=("png", "pdf"),
        )
    return fig, ax


def sup_figure_3a_ccf_coronal(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
    ax=None,
) -> tuple:
    """Supplementary figure for 3A: Sagittal and Coronal views of LC-NE cells by slicing.

    Args:
        filter_query: pandas query string to filter the metadata. If None, uses default.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """

    fig, ax = generate_ccf_plot(df_meta, filter_query, view="coronal", ax=ax)

    if if_save_figure:
        save_figure(
            fig=fig,
            filename="sup_fig_3a_ccf_sagittal_coronal_by_slicing",
            dpi=300,
            formats=("png", "pdf"),
            bbox_inches="tight",
        )
    return fig, ax


def figure_3a_ycoord_violin(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
    ax=None,
) -> tuple:
    """
    Generate and save violin plot for Y coordinate grouped by injection region.

    Args:
        df_meta: DataFrame containing metadata.
        filter_query: Optional pandas query string to filter the metadata.
        if_save_figure: Whether to save the figure to file.

    Returns:
        (fig, ax): Matplotlib figure and axes, or (None, None) if columns missing.
    """
    # Apply filter if provided
    if filter_query:
        df_meta = df_meta.query(filter_query)

    fig, ax = generate_violin_plot(
        df_to_use=df_meta,
        y_col="y",
        color_col="injection region",
        color_palette_dict=REGION_COLOR_MAPPER,
        ax=ax,
    )
    # Revert y-axis
    ax.invert_yaxis()
    ax.set_ylabel("Dorsal-ventral (μm)")
    ax.set_xlabel("")

    if if_save_figure:
        save_figure(
            fig,
            filename="fig_3a_violinplot_ycoord_by_injection_region",
            dpi=300,
            formats=("png", "pdf"),
        )
        print("Figure saved as fig_3a_violinplot_ycoord_by_injection_region.png/.pdf")
    return fig, ax


if __name__ == "__main__":
    # --- Fig 3a. Sagittal view of LC-NE cells colored by projection ---
    logger.info("Loading metadata...")
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    logger.info(f"Loaded metadata with shape: {df_meta.shape}")

    from LCNE_patchseq_analysis.figures import GLOBAL_FILTER

    figure_3a_ccf_sagittal(df_meta, GLOBAL_FILTER)
    sup_figure_3a_ccf_coronal(df_meta, GLOBAL_FILTER)
    figure_3a_ycoord_violin(df_meta, GLOBAL_FILTER)
