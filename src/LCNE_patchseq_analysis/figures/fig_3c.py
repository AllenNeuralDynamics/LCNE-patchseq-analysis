import pandas as pd

import matplotlib.pyplot as plt

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata

from LCNE_patchseq_analysis.figures.util import save_figure, generate_scatter_plot
from LCNE_patchseq_analysis.figures import DEFAULT_EPHYS_FEATURES


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
        show_marginal_y=True,
        marginal_kind="kde",
        ax=ax
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

    # Get ANOVA results
    from LCNE_patchseq_analysis.population_analysis.anova import anova_selected_ephys_features
    df_anova = anova_selected_ephys_features(df_meta, filter_query=filter_query)

    # Generate figures
    n_features = len(DEFAULT_EPHYS_FEATURES)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3.5), sharex=True,
        gridspec_kw={"wspace": 0.3, "hspace": 0.5}
    )

    for i, feature in enumerate(DEFAULT_EPHYS_FEATURES):
        col_name, feature_name = list(feature.items())[0]

        # Get p-value and adjusted p-value
        p_val_projection = df_anova.query(f'feature == "{col_name}" and term.str.contains("injection region")')["p"].values[0]
        p_adj_projection = df_anova.query(f'feature == "{col_name}" and term.str.contains("injection region")')["p_adj"].values[0]
        p_val_dv = df_anova.query(f'feature == "{col_name}" and term.str.contains("y")')["p"].values[0]
        p_adj_dv = df_anova.query(f'feature == "{col_name}" and term.str.contains("y")')["p_adj"].values[0]

        ax = axes[i // n_cols, i % n_cols]
        generate_scatter_plot(
            df=df_meta,
            y_col=col_name,
            x_col="y",
            color_col="injection region",
            color_palette=REGION_COLOR_MAPPER,
            plot_linear_regression=True,
            show_marginal_y=True,
            marginal_kind="kde",
            ax=ax
        )
        # Determine significance (any adjusted p-value < 0.05)
        sig_projection = pd.notna(p_adj_projection) and p_adj_projection < 0.05
        sig_dv = pd.notna(p_adj_dv) and p_adj_dv < 0.05
        if sig_projection and sig_dv:
            title_color = "darkred"  # both significant
        elif sig_projection:
            title_color = "royalblue"  # projection significant only
        elif sig_dv:
            title_color = "darkgreen"  # D-V significant only
        else:
            title_color = "black"  # neither significant

        # Compose multi-line styled header: feature name (bold) + stats (smaller)
        ax.set_title("")  # clear default title handling
        ax.text(
            0.5, 1.2, f"{feature_name}", transform=ax.transAxes,
            ha="center", va="bottom", fontsize=12, fontweight="bold", color=title_color,
        )
        stats_text = (
            f"Projection: p={p_val_projection:.2g} (adj={p_adj_projection:.2g})\n"
            f"D-V: p={p_val_dv:.2g} (adj={p_adj_dv:.2g})"
        )
        ax.text(
            0.05, 1.15, stats_text, transform=ax.transAxes,
            ha="left", va="top", fontsize=9, color=title_color,
        )
        ax.set_ylabel("")
        ax.set_xlabel("Dorsal-ventral (μm)")
        ax.legend_.remove()

    if if_save_figure:
        save_figure(fig, filename="sup_fig_3c_all_ipfx_features", dpi=300, formats=("png", "pdf"))
        print("Figure saved as sup_fig_3c_all_ipfx_features.png/.pdf")
    return fig, ax


if __name__ == "__main__":
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    from LCNE_patchseq_analysis.figures import GLOBAL_FILTER
    sup_figure_3c_all_ipfx_features(df_meta, GLOBAL_FILTER)
