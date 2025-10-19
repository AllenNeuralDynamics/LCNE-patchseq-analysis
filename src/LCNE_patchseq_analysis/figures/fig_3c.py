import matplotlib.pyplot as plt
import pandas as pd

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures import DEFAULT_EPHYS_FEATURES
from LCNE_patchseq_analysis.figures.util import generate_scatter_plot, save_figure
from LCNE_patchseq_analysis.population_analysis.anova import anova_features


def _generate_multi_feature_scatter_plots(
    df_meta: pd.DataFrame,
    features: list,
    df_anova: pd.DataFrame,
    filename: str,
    feature_name_mapper=None,
    if_save_figure: bool = True,
    n_cols: int = 5,
):
    """
    Abstract function to generate scatter plots for multiple features vs anatomical y coordinate.

    Args:
        df_meta: DataFrame containing metadata.
        features: List of feature column names to plot.
                  For ephys features, this should be a list of dicts like
                  [{"col_name": "display_name"}].
                  For gene/morphology, this should be a list of column names.
        df_anova: DataFrame containing ANOVA results.
        filename: Base filename for saving the figure.
        feature_name_mapper: Optional function to map column names to display names.
                             If None and features is a list of dicts,
                             uses the dict values.
                             If None and features is a list of strings, strips
                               prefix from column names.
        filter_query: Optional pandas query string to filter the metadata.
        if_save_figure: Whether to save the figure.
        n_cols: Number of columns in the subplot grid.

    Returns:
        (fig, axes): Matplotlib figure and axes array.
    """

    # Generate figures
    n_features = len(features)
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(n_cols * 4, n_rows * 3.5),
        sharex=True,
        gridspec_kw={"wspace": 0.3, "hspace": 0.5},
    )

    # Ensure axes is 2D array for consistent indexing
    if n_rows == 1 and n_cols == 1:
        axes = [[axes]]
    elif n_rows == 1:
        axes = [axes]
    elif n_cols == 1:
        axes = [[ax] for ax in axes]

    for i, feature in enumerate(features):
        # Handle different feature formats
        if isinstance(feature, dict):
            col_name, feature_name = list(feature.items())[0]
        else:
            col_name = feature
            if feature_name_mapper:
                feature_name = feature_name_mapper(col_name)
            else:
                # Default: strip common prefixes
                feature_name = (
                    col_name.replace("gene_", "").replace("morphology_", "").replace("ipfx_", "")
                )

        # Get p-value and adjusted p-value
        p_val_projection = df_anova.query(
            f'feature == "{col_name}" and term.str.contains("injection region")'
        )["p"].values[0]
        p_adj_projection = df_anova.query(
            f'feature == "{col_name}" and term.str.contains("injection region")'
        )["p_adj"].values[0]
        p_val_dv = df_anova.query(f'feature == "{col_name}" and term.str.contains("y")')[
            "p"
        ].values[0]
        p_adj_dv = df_anova.query(f'feature == "{col_name}" and term.str.contains("y")')[
            "p_adj"
        ].values[0]

        ax = axes[i // n_cols][i % n_cols]
        generate_scatter_plot(
            df=df_meta,
            y_col=col_name,
            x_col="y",
            color_col="injection region",
            color_palette=REGION_COLOR_MAPPER,
            plot_linear_regression=True,
            show_marginal_y=True,
            marginal_kind="kde",
            ax=ax,
        )

        # Compose multi-line styled header: feature name (bold) + stats (smaller)
        ax.set_title("")  # clear default title handling
        ax.text(
            0.5,
            1.2,
            f"{feature_name}",
            transform=ax.transAxes,
            ha="center",
            va="bottom",
            fontsize=12,
            fontweight="bold",
            color="royalblue" if (p_adj_projection < 0.05 or p_adj_dv < 0.05) else "black",
        )

        ax.text(
            0.05,
            1.15,
            f"Projection: p={p_val_projection:.2g} (adj={p_adj_projection:.2g})\n",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="royalblue" if p_adj_projection < 0.05 else "black",
        )
        ax.text(
            0.05,
            1.05,
            f"D-V: p={p_val_dv:.2g} (adj={p_adj_dv:.2g})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=9,
            color="royalblue" if p_adj_dv < 0.05 else "black",
        )
        ax.set_ylabel("")
        ax.set_xlabel("Dorsal-ventral (μm)")
        if ax.legend_:
            ax.legend_.remove()

    # Hide empty subplots
    for i in range(n_features, n_rows * n_cols):
        axes[i // n_cols][i % n_cols].set_visible(False)

    if if_save_figure:
        save_figure(fig, filename=filename, dpi=300, formats=("png", "pdf"))
        print(f"Figure saved as {filename}.png/.pdf")

    return fig, axes


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
        show_marginal_y=True,
        marginal_kind="kde",
        ax=ax,
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
        show_marginal_y=True,
        marginal_kind="kde",
        ax=ax,
    )

    ax.set_xlabel("Dorsal-ventral (μm)")
    ax.set_ylabel("Latency to first spike\nat rheobase (s)")

    if if_save_figure:
        save_figure(fig, filename="fig_3c_violinplot_ipfx_latency", dpi=300, formats=("png", "pdf"))
        print("Figure saved as fig_3c_violinplot_ipfx_latency.png/.pdf")
    return fig, ax


def sup_figure_3c_all_ipfx_features(
    df_meta: pd.DataFrame, filter_query: str | None = None, if_save_figure: bool = True, ax=None
):
    """
    Generate and save scatter plots for all ipfx features vs anatomical y coordinate.
    """

    # Apply filter if provided
    if filter_query:
        df_meta = df_meta.query(filter_query).copy()

    ephys_features = [list(col.keys())[0] for col in DEFAULT_EPHYS_FEATURES]

    # Get ANOVA results
    df_anova = anova_features(
        df_meta,
        features=ephys_features,
        cat_col="injection region",
        cont_col="y",
        adjust_p=True,
        anova_typ=2,
    )

    return _generate_multi_feature_scatter_plots(
        df_meta=df_meta,
        features=ephys_features,
        df_anova=df_anova,
        filename="sup_fig_3c_all_ipfx_features",
        if_save_figure=if_save_figure,
    )


def sup_figure_3b_all_gene_features(
    df_meta: pd.DataFrame, filter_query: str | None = None, if_save_figure: bool = True, ax=None
):
    """
    Generate and save scatter plots for all gene features vs anatomical y coordinate.
    """

    # Apply filter if provided
    if filter_query:
        df_meta = df_meta.query(filter_query).copy()

    # Get ANOVA results
    gene_features = [
        col for col in df_meta.columns if col.startswith("gene_") and "RNA_QC" not in col
    ]
    df_anova = anova_features(
        df_meta,
        features=gene_features,
        cat_col="injection region",
        cont_col="y",
        adjust_p=True,
        anova_typ=2,
    )

    return _generate_multi_feature_scatter_plots(
        df_meta=df_meta,
        features=gene_features,
        df_anova=df_anova,
        filename="sup_fig_3b_all_gene_features",
        if_save_figure=if_save_figure,
    )


def sup_figure_3d_morphology(
    df_meta: pd.DataFrame, filter_query: str | None = None, if_save_figure: bool = True, ax=None
):
    """
    Generate and save scatter plots for all morphology features vs anatomical y coordinate.
    """

    # Apply filter if provided
    if filter_query:
        df_meta = df_meta.query(filter_query).copy()

    # Get all morphology columns
    morphology_features = [col for col in df_meta.columns if col.startswith("morphology_")]

    # Get ANOVA results
    df_anova = anova_features(
        df_meta,
        features=morphology_features,
        cat_col="injection region",
        cont_col="y",
        adjust_p=True,
        anova_typ=2,
    )

    return _generate_multi_feature_scatter_plots(
        df_meta=df_meta,
        features=morphology_features,
        df_anova=df_anova,
        filename="sup_fig_3d_morphology",
        if_save_figure=if_save_figure,
    )


if __name__ == "__main__":
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    from LCNE_patchseq_analysis.figures import GLOBAL_FILTER

    sup_figure_3c_all_ipfx_features(df_meta, GLOBAL_FILTER)
