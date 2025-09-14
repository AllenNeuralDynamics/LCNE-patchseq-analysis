
#!/usr/bin/env python3
"""
Figure 3A: LC Mesh with Filtered Data Points (Sagittal or Coronal View)

This script creates a publication-ready figure showing the locus coeruleus mesh
with filtered patch-seq data points overlaid in sagittal or coronal view.

Default filter (used in __main__): jem-status_reporter == 'Positive' & injection region != 'Non-Retro' & != 'Thalamus'
"""

import logging
import os

import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.figure import Figure

import numpy as np
import pandas as pd

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.mesh import plot_mesh
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3
from LCNE_patchseq_analysis.figures.util import save_figure


# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
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
        
    # Legend: one entry per region
    from matplotlib.lines import Line2D
    unique_regions = df_filtered["injection region"].unique()
    legend_elements = []
    for region in unique_regions:
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

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Note: do not save or show here; let the caller decide.

    # Print summary statistics
    logger.info("\nSummary statistics:")
    logger.info(f"- Total filtered cells: {len(df_filtered)}")

    return fig, ax



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


if __name__ == "__main__":
    # --- Fig 3a. Sagittal view of LC-NE cells colored by projection ---
    logger.info("Loading metadata...")
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    logger.info(f"Loaded metadata with shape: {df_meta.shape}")

    from LCNE_patchseq_analysis.figures import global_filter
    figure_3a_ccf_sagittal(df_meta, global_filter)
    sup_figure_3a_ccf_coronal(df_meta, global_filter)

