import logging

import matplotlib.pyplot as plt
import pandas as pd

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.figures.util import generate_scatter_plot, save_figure

logger = logging.getLogger(__name__)

# All imputation panels grouped by row: (y_col, ylabel, title, kwargs)
# Row 0: pseudocluster variants (old first, then new hyperparameters)
# Row 1: MERFISH DV variants (old first, then new hyperparameters)
_IMP_PANELS = [
    # --- Row 0: pseudoclusters ---
    (
        "gene_imp_pseudoclusters (log_normed)",
        "Imputed pseudocluster\nfrom scRNA-seq",
        "Pseudoclusters (old)",
        {},
    ),
    (
        "gene_imp_pseudoclusters_gaussian_sigma0p1_k100 (log_normed)",
        "Imputed pseudocluster",
        "Pseudoclusters (gaussian_sigma0p1_k100)",
        {},
    ),
    (
        "gene_imp_pseudoclusters_gaussian_sigma1_k100 (log_normed)",
        "Imputed pseudocluster",
        "Pseudoclusters (gaussian_sigma1_k100)",
        {},
    ),
    (
        "gene_imp_pseudoclusters_softmax_tau0p1_k100 (log_normed)",
        "Imputed pseudocluster",
        "Pseudoclusters (softmax_tau0p1_k100)",
        {},
    ),
    (
        "gene_imp_pseudoclusters_softmax_tau0p5_k100 (log_normed)",
        "Imputed pseudocluster",
        "Pseudoclusters (softmax_tau0p5_k100)",
        {},
    ),
    # --- Row 1: MERFISH DV ---
    (
        "gene_imp_DV (log_normed)",
        "Imputed DV from MERFISH (μm)",
        "MERFISH DV (old)",
        {"if_trim": False, "if_same_xy": True},
    ),
    (
        "gene_imp_DV_gaussian_sigma0p1_k100 (log_normed)",
        "Imputed DV from MERFISH (μm)",
        "MERFISH DV (gaussian_sigma0p1_k100)",
        {"if_trim": False, "if_same_xy": True},
    ),
]

_N_COLS = 5  # pseudocluster row has 5 panels → drives grid width


def _scatter_imp(
    df_meta: pd.DataFrame,
    y_col: str,
    ylabel: str,
    plot_linear_regression: bool = True,
    ax=None,
    **kwargs,
):
    """Generic helper: scatter of one imputed column vs anatomical y (DV, µm)."""
    fig, ax = generate_scatter_plot(
        df=df_meta,
        y_col=y_col,
        x_col="y",
        color_col="injection region",
        color_palette=REGION_COLOR_MAPPER,
        plot_linear_regression=plot_linear_regression,
        regression_type="type1",
        show_marginal_y=False,
        ax=ax,
        **kwargs,
    )
    ax.set_xlabel("Dorsal-ventral (μm)")
    ax.set_ylabel(ylabel)
    return fig, ax


def imputed_scRNAseq(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
    plot_linear_regression: bool = True,
    ax=None,
):
    """Figure 3B: Scatter of imputed scRNAseq pseudoclusters vs anatomical y coordinate."""
    if filter_query:
        df_meta = df_meta.query(filter_query)
    y_col, ylabel, _, kwargs = _IMP_PANELS[0]
    fig, ax = _scatter_imp(df_meta, y_col, ylabel, plot_linear_regression, ax, **kwargs)
    if if_save_figure:
        save_figure(
            fig, filename="fig_3b_scatter_imp_pseudoclusters_vs_y", dpi=300, formats=("png", "svg")
        )
    return fig, ax


def imputed_MERFISH(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
    plot_linear_regression: bool = True,
    ax=None,
):
    """Figure 3B: Scatter of imputed MERFISH DV coordinate vs anatomical y coordinate."""
    if filter_query:
        df_meta = df_meta.query(filter_query).copy()
    y_col, ylabel, _, kwargs = _IMP_PANELS[1]
    fig, ax = _scatter_imp(df_meta, y_col, ylabel, plot_linear_regression, ax, **kwargs)
    if if_save_figure:
        save_figure(
            fig, filename="fig_3b_scatter_imp_MERFISH_vs_y", dpi=300, formats=("png", "svg")
        )
    return fig, ax


def main_imputation(
    df_meta: pd.DataFrame,
    filter_query: str | None = None,
    if_save_figure: bool = True,
    subplot_size: float = 4.0,
):
    """Generate all imputation scatters (old + new hyperparameters) in a 2×5 grid.

    Row 0: pseudocluster variants (5 panels).
    Row 1: MERFISH DV variants (2 panels, remaining cells hidden).
    All subplots are square.
    """
    if filter_query:
        df_meta = df_meta.query(filter_query).copy()

    n_rows, n_cols = 2, _N_COLS
    figsize = (n_cols * subplot_size, n_rows * subplot_size)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)

    for i, (y_col, ylabel, title, kwargs) in enumerate(_IMP_PANELS):
        row, col = divmod(i, n_cols)
        ax = axes[row, col]
        if y_col not in df_meta.columns:
            ax.set_visible(False)
            continue
        _, ax = _scatter_imp(df_meta, y_col, ylabel, ax=ax, **kwargs)
        ax.set_title(title, fontsize=9)
        ax.set_box_aspect(1)  # force square subplot
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    # Hide unused cells in row 1
    n_used_row1 = len(_IMP_PANELS) - n_cols  # panels that spill into row 1
    for col in range(n_used_row1, n_cols):
        axes[1, col].set_visible(False)

    fig.tight_layout()

    if if_save_figure:
        save_figure(
            fig, filename="main_imputation", dpi=300, formats=("png", "svg"), bbox_inches="tight"
        )

    return fig, axes


if __name__ == "__main__":
    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
    from LCNE_patchseq_analysis.figures import GENE_FILTER

    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    main_imputation(df_meta, GENE_FILTER)
