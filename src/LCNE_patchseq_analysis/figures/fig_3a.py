

import pandas as pd
import logging
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures.util import create_violin_plot_matplotlib, save_figure, plot_in_ccf
from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.figures import sort_region


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()


def figure_3a_ccf_sagittal(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
) -> tuple:
    """Deprecated wrapper around plot_in_ccf with optional filter and angle.

    Args:
        filter_query: pandas query string to filter the metadata. If None, uses default.
        slicing_angle: 'sagittal' or 'coronal'. Defaults to 'sagittal'.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """

    fig, ax = plot_in_ccf(df_meta, filter_query, view="sagittal")

    if if_save_figure:
        save_figure(
            fig=fig,
            filename="fig_3a_ccf_sagittal_by_projection",
            dpi=300,
            formats=("png", "pdf"),
        )
    return fig, ax


def sup_figure_3a_ccf_coronal(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
) -> tuple:
    """Supplementary figure for 3A: Sagittal and Coronal views of LC-NE cells by slicing.

    Args:
        filter_query: pandas query string to filter the metadata. If None, uses default.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """

    fig, ax = plot_in_ccf(df_meta, filter_query, view="coronal")

    if if_save_figure:
        save_figure(
            fig=fig,
            filename="sup_fig_3a_ccf_sagittal_coronal_by_slicing",
            dpi=300,
            formats=("png", "pdf"),
        )
    return fig, ax



def figure_3a_ycoord_violin(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True
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

    fig, ax = create_violin_plot_matplotlib(
        df_to_use=df_meta,
        y_col="y",
        color_col="injection region",
        color_palette_dict=REGION_COLOR_MAPPER,
        font_size=12
    )
    # Revert y-axis
    ax.invert_yaxis()

    if if_save_figure:
        save_figure(fig, filename="fig_3a_violinplot_ycoord_by_injection_region", dpi=300, formats=("png", "pdf"))
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

