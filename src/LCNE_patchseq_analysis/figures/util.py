"""Plotting utilities for LCNE patchseq analysis figures."""

import os
import logging
from typing import Mapping, Sequence

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib import colors as mcolors
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1 import make_axes_locatable

from LCNE_patchseq_analysis.data_util.mesh import plot_mesh
from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3
from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.figures import sort_region

logger = logging.getLogger(__name__)


def generate_scatter_plot(
    df: pd.DataFrame,
    y_col: str,
    x_col: str,
    color_col: str,
    color_palette: Mapping[str, str] | None = None,
    plot_linear_regression: bool = True,
    point_size: int = 40,
    alpha: float = 0.8,
    figsize: tuple = (4, 4),
    show_marginal: bool = False,
    marginal_kind: str = "kde",
    marginal_size: str = "25%",
    marginal_pad: float = 0.05,
    ax=None,
):
    """Generic scatter plot utility for Figure 3B style.

    Args:
        df: Input dataframe.
        y_col: Column for y axis.
        x_col: Column for x axis.
        color_col: Column that defines groups / colors.
        color_palette: Mapping from group value to color. If None uses REGION_COLOR_MAPPER for injection regions.
        plot_linear_regression: If True overlay OLS line across all points and annotate p and R^2.
        point_size: Scatter marker size.
        alpha: Point transparency.
        figsize: Figure size.
        show_marginal: If True, draw a marginal distribution of y values on the right side split by color_col.
        marginal_kind: One of {'kde','hist'} selecting marginal distribution type.
        marginal_size: Width of marginal axes (passed to axes_grid1 append_axes size argument).
        marginal_pad: Padding between main and marginal axes.
    Returns:
        (fig, ax) main axes (a reference to marginal axes is stored at ax.marginal_ax if created).
    """
    if color_palette is None and color_col.lower() == "injection region":
        color_palette = REGION_COLOR_MAPPER

    # Drop rows missing required columns
    required_cols = [y_col, x_col, color_col]
    df_plot = df.dropna(subset=required_cols).copy()
    if df_plot.empty:
        raise ValueError("No data left after dropping NA for required columns")

    # Determine plotting order if injection region
    hue_order: Sequence[str] | None = None
    if color_col.lower() == "injection region":
        unique_regions = df_plot[color_col].dropna().unique().tolist()
        hue_order = sort_region(unique_regions)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.figure
    sns.scatterplot(
        data=df_plot,
        x=x_col,
        y=y_col,
        hue=color_col,
        palette=color_palette,
        hue_order=hue_order,
        s=point_size,
        alpha=alpha,
        edgecolor="k",
        linewidth=0.3,
        ax=ax,
    )

    # Optional marginal distribution panel (right side) sharing y-axis
    if show_marginal:
        try:
            groups = hue_order if hue_order is not None else sorted(df_plot[color_col].dropna().unique())
            # Build color lookup
            if color_palette is None:
                # Fall back to seaborn palette if none provided
                palette_lut = dict(zip(groups, sns.color_palette(n_colors=len(groups))))
            else:
                palette_lut = {g: color_palette.get(g, "gray") for g in groups}

            divider = make_axes_locatable(ax)
            ax_marg = divider.append_axes("right", size=marginal_size, pad=marginal_pad, sharey=ax)
            for g in groups:
                sub = df_plot[df_plot[color_col] == g]
                if sub.empty:
                    continue
                color = palette_lut.get(g, "gray")
                if marginal_kind == "kde":
                    try:
                        sns.kdeplot(
                            data=sub,
                            y=y_col,
                            ax=ax_marg,
                            color=color,
                            fill=False,
                            linewidth=1.0,
                            common_norm=False,
                            cut=1,
                        )
                    except Exception as e:  # noqa: BLE001
                        logger.warning(f"KDE marginal failed for group {g}: {e}")
                elif marginal_kind == "hist":
                    sns.histplot(
                        data=sub,
                        y=y_col,
                        ax=ax_marg,
                        color=color,
                        element="step",
                        fill=False,
                        bins=20,
                        stat="density",
                        alpha=0.9,
                    )
                else:
                    logger.warning(f"Unsupported marginal_kind '{marginal_kind}'. Skipping marginal plot.")
                    break
            # Cosmetic cleanup
            ax_marg.set_xlabel("kde" if marginal_kind == "kde" else "Count")
            ax_marg.set_ylabel("")
            ax_marg.set_xticks([])
            # Remove only marginal axis y ticks / labels (keep main axis intact)
            ax_marg.tick_params(axis="y", left=False, labelleft=False)
            sns.despine(ax=ax_marg, left=True, bottom=True)
            # Attach for downstream access
            ax.marginal_ax = ax_marg  # type: ignore[attr-defined]
        except Exception as e:  # noqa: BLE001
            logger.warning(f"Failed to create marginal axes: {e}")

    # Optional regression across all points (ignoring grouping)
    if plot_linear_regression:
        try:
            from scipy.stats import linregress  # type: ignore

            res = linregress(df_plot[x_col], df_plot[y_col])
            x_vals = pd.Series(sorted(df_plot[x_col].values))
            y_fit = res.intercept + res.slope * x_vals
            ax.plot(x_vals, y_fit, color="black", linewidth=1.2, zorder=5, label="Linear fit")
            # Annotation: p-value and R^2
            r_squared = res.rvalue ** 2
            annotation = f"p={res.pvalue:.2e}\nR={np.sqrt(r_squared):.2f}"
            ax.text(
                0.98,
                0.85,
                annotation,
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5),
                fontsize=9,
            )
        except Exception as e:  # noqa: BLE001 - lightweight handling, controlled scope
            logger.warning(f"Linear regression failed: {e}")

    ax.set_xlabel(x_col)
    ax.set_ylabel(y_col)
    ax.legend(title=color_col, loc="best")
    sns.despine(trim=True, ax=ax)
    # fig.tight_layout()
    return fig, ax

def generate_ccf_plot(
    df_meta: pd.DataFrame,
    filter_query: str | None,
    view: str,
    ax=None,
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
        x_key, y_key, mesh_direction, x_label = "x", "y", "sagittal", "Anterior-posterior (μm)"
    elif view == "coronal":
        x_key, y_key, mesh_direction, x_label = "z", "y", "coronal", "Left-right (μm)"
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


    # Create the plot with matplotlib backend only if ax is None
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 5))
    else:
        fig = ax.figure

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
    ax.legend(handles=legend_elements, bbox_to_anchor=(1.05, 1), loc='upper left')

    # Add labels and title
    ax.set_xlabel(x_label)
    ax.set_ylabel("Dorsal-ventral (μm)")
    ax.set_title(
        f"LC Mesh with Filtered Data Points ({view.capitalize()} View)\n"
        + f"Filter: {filter_query}\n"
        + f"n = {len(df_filtered)} cells",
    )

    # Ensure equal aspect ratio is maintained
    ax.set_aspect("equal")

    # plt.tight_layout()

    # Print summary statistics
    logger.info("\nSummary statistics:")
    logger.info(f"- Total filtered cells: {len(df_filtered)}")

    return fig, ax



def generate_violin_plot(
    df_to_use: pd.DataFrame,
    y_col: str,
    color_col: str,
    color_palette_dict: dict,
    ax=None
):
    """
    Create a violin plot to compare data distributions across groups using matplotlib/seaborn.

    Args:
        df_to_use: DataFrame containing the data.
        y_col: Column name for y-axis variable.
        color_col: Column name for color grouping.
        color_palette_dict: Optional dict mapping group names to colors.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """

    if ax is None:
        fig, ax = plt.subplots(figsize=(5, 4))
    else:
        fig = ax.figure
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
        f"{group}\n(n={group_counts.get(group, 0)})"
        for group in groups_order
    ]
    logger.info("Group counts (non-NA): " + ", ".join(
        [f"{group}: {group_counts.get(group, 0)}" for group in groups_order]
    ))
    ax.set_xticks(range(len(groups_order)))
    ax.set_xticklabels(group_labels_with_counts, rotation=30, ha='right')
    ax.set_ylabel(y_col)
    ax.set_xlabel(color_col)
    sns.despine(trim=True, ax=ax)
    return fig, ax


def save_figure(
    fig: Figure,
    output_dir: str | None = None,
    filename: str = "plot",
    dpi: int = 300,
    formats: tuple[str, ...] = ("png", "pdf"),
    **kwargs
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
            facecolor="white",
            edgecolor="none",
            **kwargs
        )
        logger.info(f"Figure saved as: {out_path}")
        saved_paths.append(out_path)

    return saved_paths
