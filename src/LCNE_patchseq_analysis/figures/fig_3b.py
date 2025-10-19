import logging

import pandas as pd

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.figures.util import generate_scatter_plot, save_figure

logger = logging.getLogger(__name__)


def figure_3b_imputed_scRNAseq(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
    plot_linear_regression: bool = True,
    ax=None,
):
    """Figure 3B: Scatter of imputed scRNAseq pseudoclusters vs anatomical y coordinate.

    Assumptions:
            - x-axis coordinate column is named 'y'
            (consistent with figure 3a usage of anatomical y).
            - Imputed pseudocluster column name here assumed
            'gene_imp_pseudoclusters (log_normed)'.
    """
    if filter_query:
        df_meta = df_meta.query(filter_query)

    fig, ax = generate_scatter_plot(
        df=df_meta,
        y_col="gene_imp_pseudoclusters (log_normed)",
        x_col="y",
        color_col="injection region",
        color_palette=REGION_COLOR_MAPPER,
        plot_linear_regression=plot_linear_regression,
        show_marginal_y=True,
        marginal_kind="kde",
        ax=ax,
    )
    # ax.invert_yaxis()
    ax.set_xlabel("Dorsal-ventral (μm)")
    ax.set_ylabel("Imputed pseudocluster\nfrom scRNA-seq")

    if if_save_figure:
        save_figure(
            fig,
            filename="fig_3b_scatter_imp_pseudoclusters_vs_y",
            dpi=300,
            formats=("png", "pdf"),
        )
    return fig, ax


def figure_3b_imputed_MERFISH(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
    plot_linear_regression: bool = True,
    ax=None,
):
    """Figure 3B: Scatter of imputed MERFISH pseudoclusters vs anatomical y coordinate.

    Assumptions:
            - x-axis coordinate column is named 'y'
            (consistent with figure 3a usage of anatomical y).
            - Imputed pseudocluster column name here assumed
              'gene_imp_pseudoclusters (log_normed)'.
    """
    if filter_query:
        df_meta = df_meta.query(filter_query).copy()

    df_meta["gene_imp_DV in um (log_normed)"] = (
        df_meta["gene_imp_DV (log_normed)"] * 25
    )  # TODO: confirm with Shuonan
    fig, ax = generate_scatter_plot(
        df=df_meta,
        y_col="gene_imp_DV in um (log_normed)",
        x_col="y",
        color_col="injection region",
        color_palette=REGION_COLOR_MAPPER,
        plot_linear_regression=plot_linear_regression,
        show_marginal_y=True,
        marginal_kind="kde",
        if_trim=False,
        if_same_xy=True,
        ax=ax,
    )
    ax.set_xlabel("Dorsal-ventral (μm)")
    ax.set_ylabel("Imputed dorsal-ventral\nfrom MERFISH (μm)")

    if if_save_figure:
        save_figure(
            fig,
            filename="fig_3b_scatter_imp_MERFISH_vs_y",
            dpi=300,
            formats=("png", "pdf"),
        )
    return fig, ax


if __name__ == "__main__":
    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
    from LCNE_patchseq_analysis.figures import GENE_FILTER

    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)

    # For gene data, apply additional filtering
    figure_3b_imputed_scRNAseq(df_meta, GENE_FILTER)
    figure_3b_imputed_MERFISH(df_meta, GENE_FILTER)
