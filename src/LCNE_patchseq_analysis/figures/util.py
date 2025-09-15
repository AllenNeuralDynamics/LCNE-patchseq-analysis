
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D

import seaborn as sns

from LCNE_patchseq_analysis.data_util.mesh import plot_mesh
from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3
from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.figures import sort_region


logger = logging.getLogger()

def plot_in_ccf(
    df_meta: pd.DataFrame,
    filter_query: str | None,
    view: str,
) -> tuple:
    """Generate LC mesh projection with filtered neurons in sagittal or coronal view.

    Args:
        filter_query: A pandas query string to filter metadata dataframe.
        view: Which plane to plot. Accepts 'sagittal' or 'coronal'.

    Returns:
        (fig, ax): The matplotlib figure/axes.
    """


    view = (view or "").strip().lower()
    if view == "sagittal":
        x_key, y_key, mesh_direction, x_label = "x", "y", "sagittal", "X Coordinate (μm)"
    elif view == "coronal":
        x_key, y_key, mesh_direction, x_label = "z", "y", "coronal", "Z Coordinate (μm)"
    else:
        raise ValueError(f"Invalid view '{view}'. Use 'sagittal' or 'coronal'.")


    # Apply the specified filter
    if filter_query:
        df_filtered = df_meta.query(filter_query)
    else:
        df_filtered = df_meta

    logger.info(f"Total cells after filtering: {len(df_filtered)}")
    logger.info("Injection regions in filtered data:")
    region_counts = df_filtered["injection region"].value_counts()
    for region, count in region_counts.items():
        logger.info(f"  {region}: {count}")

    # Load the LC mesh
    logger.info("Loading LC mesh...")
    mesh = load_mesh_from_s3()

    # Create the plot with matplotlib backend
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)

    # Plot the mesh first according to selected view
    plot_mesh(ax, mesh, direction=mesh_direction, meshcol="lightgray")

    logger.info("Color mapping & style assignment:")
    for idx, row in df_filtered.iterrows():
        region = row["injection region"]
        color_key = region if region in REGION_COLOR_MAPPER else region.lower()
        color = REGION_COLOR_MAPPER.get(color_key, "gray")
        # Slicing plane match?
        match = row["slicing plane"].lower() == view
        lw = 3 if match else 0.5
        ls = 'solid' if match else 'dotted'
        ax.scatter(
            row[x_key], row[y_key],
            c=[mcolors.to_rgba(color, 1.0)],
            s=60,
            edgecolors='black',
            linestyle=ls,
            label=None,
            alpha=0.6
        )
        
    unique_regions = df_filtered["injection region"].unique()
    sorted_regions = sort_region(unique_regions)

    # Sort df_filtered by injection region according to sorted_regions
    df_filtered = df_filtered.set_index('injection region').loc[sorted_regions].reset_index()

    legend_elements = []
    for region in sorted_regions:
        color_key = region if region in REGION_COLOR_MAPPER else region.lower()
        color = REGION_COLOR_MAPPER.get(color_key, "gray")
        label_text = f"{region} (n={sum(df_filtered['injection region']==region)})"
        legend_elements.append(
            Line2D(
                [0], [0],
                marker='o',
                color='black',
                markerfacecolor=color,
                markeredgecolor='black',
                markersize=6,
                linestyle='None',
                label=label_text,
            )
        )
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Y Coordinate (μm)", fontsize=12)
    ax.set_title(
        f"LC Mesh with Filtered Data Points ({view.capitalize()} View)\n"
        + f"Filter: {filter_query}\n"
        + f"n = {len(df_filtered)} cells",
        fontsize=10,
    )

    # Ensure equal aspect ratio is maintained
    ax.set_aspect("equal")

    plt.tight_layout()

    # Print summary statistics
    logger.info("\nSummary statistics:")
    logger.info(f"- Total filtered cells: {len(df_filtered)}")


    return fig, ax



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
        nan_count = np.sum(pd.isna(df_to_use.loc[group_mask, y_col]))
        group_nan_counts[group] = nan_count

    # Convert y_col to numeric

    plot_df[y_col] = pd.to_numeric(plot_df[y_col], errors='coerce')
    plot_df = plot_df.dropna(subset=[y_col])
    if plot_df.empty:
        return fig, ax


    # If color_col = 'injection region', sort by predefined order
    if color_col == 'injection region':
        groups_order = sort_region(plot_df[color_col].unique())
    else:
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
            ax.plot(i + 0.45, mean_val, 'o', color=group_color, markersize=5,
                    markeredgecolor='black', markeredgewidth=1, zorder=10)
            if sem_val > 0.0:
                ax.errorbar(i + 0.45, mean_val, yerr=sem_val, color='black',
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


def save_figure(
    fig: Figure,
    output_dir: str | None = None,
    filename: str = "plot",
    dpi: int = 300,
    formats: tuple[str, ...] = ("png", "pdf"),
) -> list[str]:
    """Save a matplotlib Figure with standardized naming.

    Args:
        fig: The matplotlib Figure to save.
        output_dir: Directory to save into; defaults to this script directory.
        filename: The filename without extension.
        dpi: Resolution for raster formats (e.g., PNG).
        formats: File formats to save, e.g., ("png", "pdf").

    Returns:
        List of saved file paths in the same order as formats.
    """
    if output_dir is None:
        output_dir = os.path.dirname(os.path.abspath(__file__))

    os.makedirs(output_dir, exist_ok=True)

    saved_paths: list[str] = []
    for ext in formats:
        out_path = os.path.join(output_dir, f"{filename}.{ext}")
        fig.savefig(
            out_path,
            dpi=dpi,
            bbox_inches="tight",
            facecolor="white",
            edgecolor="none",
        )
        logger.info(f"Figure saved as: {out_path}")
        saved_paths.append(out_path)

    return saved_paths
