import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures.util import save_figure


def create_violin_plot_matplotlib(
    df_to_use: pd.DataFrame,
    y_col: str,
    color_col: str,
    color_palette_dict: dict,
    font_size: int = 12
):
    """
    Create a violin plot to compare data distributions across groups using matplotlib/seaborn.

    Args:
        df_to_use: DataFrame containing the data.
        y_col: Column name for y-axis variable.
        color_col: Column name for color grouping.
        color_palette_dict: Optional dict mapping group names to colors.
        font_size: Font size for labels.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """
    fig, ax = plt.subplots(figsize=(5, 4), dpi=300)
    plot_df = df_to_use[[y_col, color_col]].dropna()
    if plot_df.empty:
        return fig, ax

    # Count number of samples per group (valid data)
    group_counts = plot_df[color_col].value_counts().to_dict()
    # Count missing data (NaN) per group from original dataframe
    group_nan_counts = {}
    for group in group_counts.keys():
        group_mask = df_to_use[color_col] == group
        nan_count = pd.isna(df_to_use.loc[group_mask, y_col]).sum()
        group_nan_counts[group] = nan_count

    # Convert y_col to numeric
    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[y_col])
    if plot_df.empty:
        return fig, ax

    groups_order = sorted(plot_df[color_col].unique())

    # Violin plot

    sns.violinplot(
        data=plot_df,
        x=color_col,
        y=y_col,
        hue=color_col,
        ax=ax,
        palette=color_palette_dict,
        inner="quart",
        alpha=0.6,
        cut=0,
        order=groups_order,
        width=0.5,
        legend=False
    )

    # Overlay raw data points
    sns.stripplot(
        data=plot_df,
        x=color_col,
        y=y_col,
        ax=ax,
        color='black',
        size=2,
        alpha=0.5,
        jitter=True,
        order=groups_order
    )

    # Plot mean ± SEM for each group
    for i, group in enumerate(groups_order):
        group_data = pd.to_numeric(plot_df[plot_df[color_col] == group][y_col], errors='coerce').dropna()
        if len(group_data) > 0:
            mean_val = float(np.mean(group_data))
            sem_val = float(np.std(group_data, ddof=1) / np.sqrt(len(group_data))) if len(group_data) > 1 else 0.0
            group_color = color_palette_dict.get(group, 'black') if color_palette_dict else 'black'
            ax.plot(i+0.45, mean_val, 'o', color=group_color, markersize=5,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)
            if sem_val > 0.0:
                ax.errorbar(i+0.45, mean_val, yerr=sem_val, color='black',
                            capsize=5, capthick=1, elinewidth=1, zorder=9, fmt='none')

    # Set x-axis labels with sample counts
    group_labels_with_counts = [
        f"{group}\n(n={group_counts.get(group, 0)}, missing {group_nan_counts.get(group, 0)})"
        for group in groups_order
    ]
    ax.set_xticks(range(len(groups_order)))
    ax.set_xticklabels(group_labels_with_counts, rotation=30, ha='right', fontsize=font_size)
    ax.set_ylabel(y_col, fontsize=font_size)
    ax.set_xlabel(color_col, fontsize=font_size)
    sns.despine(trim=True)
    plt.tight_layout()
    return fig, ax


def figure_3a_tau_comparison(
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

    # Only plot if the columns exist
    fig, ax = create_violin_plot_matplotlib(
        df_to_use=df_meta,
        y_col="ipfx_tau",
        color_col="injection region",
        color_palette_dict=REGION_COLOR_MAPPER,
        font_size=12
    )
    if if_save_figure:
        save_figure(fig, filename="fig_3b_violinplot_ipfx_tau", dpi=300, formats=("png", "pdf"))
        print("Figure saved as fig_3b_violinplot_ipfx_tau.png/.pdf")
    return fig, ax


if __name__ == "__main__":
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    from LCNE_patchseq_analysis.figures import global_filter
    figure_3a_tau_comparison(df_meta, global_filter)
