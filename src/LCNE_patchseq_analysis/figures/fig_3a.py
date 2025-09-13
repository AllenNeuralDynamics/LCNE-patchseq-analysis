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
from matplotlib.figure import Figure

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()


def plot_in_ccf(
    filter_query: str | None,
    slicing_angle: str,
) -> tuple:
    """Generate LC mesh projection with filtered neurons in sagittal or coronal view.

    Args:
        filter_query: A pandas query string to filter metadata dataframe.
        slicing_angle: Which plane to plot. Accepts 'sagittal' or 'coronal'.

    Returns:
        (fig, ax): The matplotlib figure/axes.
    """

    # Import required modules
    from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
    from LCNE_patchseq_analysis.data_util.mesh import plot_mesh
    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
    from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3

    logger.info("Loading metadata...")
    # Load metadata
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    logger.info(f"Loaded metadata with shape: {df_meta.shape}")

    angle = (slicing_angle or "").strip().lower()
    if angle == "sagittal":
        view = "sagittal"
        x_key = "x"
        y_key = "y"
        mesh_direction = "sagittal"
        x_label = "X Coordinate (μm)"
    elif angle == "coronal":
        view = "coronal"
        x_key = "z"
        y_key = "y"
        mesh_direction = "coronal"
        x_label = "Z Coordinate (μm)"
    else:
        raise ValueError(
            f"Invalid slicing_angle '{slicing_angle}'. Use 'sagittal' or 'coronal'."
        )

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

    # Get unique regions and prepare colors using REGION_COLOR_MAPPER
    unique_regions = df_filtered["injection region"].unique()
    region_colors = []

    logger.info("Color mapping:")
    for region in unique_regions:
        # Try to get color from REGION_COLOR_MAPPER, with fallback
        color_key = region if region in REGION_COLOR_MAPPER else region.lower()
        color = REGION_COLOR_MAPPER.get(color_key, "gray")
        region_colors.append(color)
        logger.info(f"  {region}: {color}")

    # Plot the filtered data points with consistent colors
    for i, region in enumerate(unique_regions):
        region_data = df_filtered[df_filtered["injection region"] == region]
        ax.scatter(
            region_data[x_key],
            region_data[y_key],
            c=region_colors[i],
            alpha=0.7,
            s=50,
            edgecolors="black",
            linewidth=0.5,
            label=(
                f"{region} (n={len(region_data)}, "
                f"{region_data[x_key].isnull().sum()} missing {x_key})"
            ),
        )

    # Add labels and title
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel("Y Coordinate (μm)", fontsize=12)
    ax.set_title(
        f"LC Mesh with Filtered Data Points ({view.capitalize()} View)\n"
        + f"Filter: {filter_query}\n"
        + f"n = {len(df_filtered)} cells",
        fontsize=10,
    )

    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left", fontsize=10)

    # Ensure equal aspect ratio is maintained
    ax.set_aspect("equal")

    # Adjust layout to prevent legend cutoff
    plt.tight_layout()

    # Note: do not save or show here; let the caller decide.

    # Print summary statistics
    logger.info("\nSummary statistics:")
    logger.info(f"- Total filtered cells: {len(df_filtered)}")

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

def figure_3a_ccf_sagittal_by_projection(
    filter_query: str | None = None,
) -> tuple:
    """Deprecated wrapper around plot_in_ccf with optional filter and angle.

    Args:
        filter_query: pandas query string to filter the metadata. If None, uses default.
        slicing_angle: 'sagittal' or 'coronal'. Defaults to 'sagittal'.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """

    fig, ax = plot_in_ccf(filter_query, slicing_angle="sagittal")

    save_figure(
        fig=fig,
        filename="fig_3a_ccf_sagittal_by_projection",
        dpi=300,
        formats=("png", "pdf"),
    )
    return fig, ax


def sup_figure_3a_ccf_sagittal_coronal_by_slicing(
        filter_query: str | None = None,
) -> tuple:
    """Supplementary figure for 3A: Sagittal and Coronal views of LC-NE cells by slicing.

    Args:
        filter_query: pandas query string to filter the metadata. If None, uses default.

    Returns:
        (fig, ax): Matplotlib figure and axes.
    """

    fig, ax = plot_in_ccf(filter_query, slicing_angle="coronal", )

    save_figure(
        fig=fig,
        filename="sup_fig_3a_ccf_sagittal_coronal_by_slicing",
        dpi=300,
        formats=("png", "pdf"),
    )
    return fig, ax


if __name__ == "__main__":
    try:
        # Defaults matching the previous behavior
        global_filter = (
            "(`jem-status_reporter` == 'Positive') & "
            "(`injection region` != 'Non-Retro') & "
            "(`injection region` != 'Thalamus')"
        )

        # --- Fig 3a. Sagittal view of LC-NE cells colored by projection ---
        # figure_3a_ccf_sagittal_by_projection(global_filter)  # Done

        sup_figure_3a_ccf_sagittal_coronal_by_slicing(global_filter)


    except Exception as e:
        logger.error(f"Error generating Figure 3A: {e}")
        raise
