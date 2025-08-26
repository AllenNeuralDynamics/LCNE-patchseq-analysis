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

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger()


def plot_in_ccf(
    filter_query: str,
    slicing_angle: str,
    if_save_fig: bool = True,
) -> tuple:
    """Generate LC mesh projection with filtered neurons in sagittal or coronal view.

    Args:
        filter_query: A pandas query string to filter metadata dataframe.
    slicing_angle: Which plane to plot. Accepts 'sagittal' or 'coronal'.
        if_save_fig: Whether to save PNG and PDF in the same folder as this script.

    Returns:
        (fig, ax, df_filtered): The matplotlib figure/axes and the filtered dataframe.
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

    # Normalize slicing angle (strict)
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
    df_filtered = df_meta.query(filter_query)

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

    # If saving is enabled, save the figure
    if if_save_fig:
        # Determine the output directory as the script's folder
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Save the figure as PNG
        output_filename = os.path.join(
            script_dir, f"fig_3a_ccf_projection_{view}.png"
        )
        plt.savefig(
            output_filename, dpi=300, bbox_inches="tight", facecolor="white", edgecolor="none"
        )
        logger.info(f"Figure saved as: {output_filename}")

        # Also save as PDF for publication
        output_pdf = os.path.join(script_dir, f"fig_3a_ccf_projection_{view}.pdf")
        plt.savefig(output_pdf, bbox_inches="tight", facecolor="white", edgecolor="none")
        logger.info(f"Figure saved as: {output_pdf}")
    else:
        # Show the plot
        plt.show()

    # Print summary statistics
    logger.info("\nSummary statistics:")
    logger.info(f"- Total filtered cells: {len(df_filtered)}")

    return fig, ax, df_filtered


def figure_3a_ccf_projection(if_save_fig: bool = True) -> tuple:
    """Deprecated: use plot_in_ccf(filter_query, slicing_angle, if_save_fig).

    This wrapper preserves the old API used in notebooks by applying the
    historical default filter and sagittal view.
    """
    logger.warning(
        "figure_3a_ccf_projection is deprecated. Use plot_in_ccf(filter_query, slicing_angle, if_save_fig)."
    )
    default_filter_query = (
        "(`jem-status_reporter` == 'Positive') & "
        "(`injection region` != 'Non-Retro') & "
        "(`injection region` != 'Thalamus')"
    )
    return plot_in_ccf(default_filter_query, "sagittal", if_save_fig=if_save_fig)


if __name__ == "__main__":
    try:
        # Defaults matching the previous behavior
        default_filter_query = (
            "(`jem-status_reporter` == 'Positive') & "
            "(`injection region` != 'Non-Retro') & "
            "(`injection region` != 'Thalamus')"
        )
        fig, ax, df_filtered = plot_in_ccf(
            default_filter_query, slicing_angle="sagittal"
        )
        print("Figure 3A generated successfully!")
    except Exception as e:
        logger.error(f"Error generating Figure 3A: {e}")
        raise
