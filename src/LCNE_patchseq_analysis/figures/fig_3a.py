#!/usr/bin/env python3
"""
Figure 3A: LC Mesh with Filtered Data Points (Sagittal View)

This script creates a publication-ready figure showing the locus coeruleus mesh
with filtered patch-seq data points overlaid in sagittal view.

Filter: jem-status_reporter == 'Positive' & injection region != 'Non-Retro' & != 'Thalamus'
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger()

def figure_3a_ccf_projection(if_save_fig: bool = True) -> tuple:
    """Main function to generate Figure 3A"""
    
    # Import required modules
    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
    from LCNE_patchseq_analysis.data_util.mesh import plot_mesh
    from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3
    from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
    
    logger.info("Loading metadata...")
    # Load metadata
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    logger.info(f"Loaded metadata with shape: {df_meta.shape}")
    
    # Apply the specified filter
    filter_query = "(`jem-status_reporter` == 'Positive') & (`injection region` != 'Non-Retro') & (`injection region` != 'Thalamus')"
    df_filtered = df_meta.query(filter_query)
    
    logger.info(f"Total cells after filtering: {len(df_filtered)}")
    logger.info("Injection regions in filtered data:")
    region_counts = df_filtered['injection region'].value_counts()
    for region, count in region_counts.items():
        logger.info(f"  {region}: {count}")
    
    # Load the LC mesh
    logger.info("Loading LC mesh...")
    mesh = load_mesh_from_s3()
    
    # Create the plot with matplotlib backend
    fig, ax = plt.subplots(figsize=(8, 5), dpi=300)
    
    # Plot the mesh first (sagittal view)
    plot_mesh(ax, mesh, direction="sagittal", meshcol="lightgray")
        
    # Get unique regions and prepare colors using REGION_COLOR_MAPPER
    unique_regions = df_filtered['injection region'].unique()
    region_colors = []
    
    logger.info("Color mapping:")
    for region in unique_regions:
        # Try to get color from REGION_COLOR_MAPPER, with fallback
        color_key = region if region in REGION_COLOR_MAPPER else region.lower()
        color = REGION_COLOR_MAPPER.get(color_key, 'gray')
        region_colors.append(color)
        logger.info(f"  {region}: {color}")
    
    # Plot the filtered data points with consistent colors
    for i, region in enumerate(unique_regions):
        region_data = df_filtered[df_filtered['injection region'] == region]
        ax.scatter(region_data['x'], region_data['y'], 
                  c=region_colors[i], alpha=0.7, s=50, 
                  edgecolors='black', linewidth=0.5,
                  label=f'{region} (n={len(region_data)})')
    
    # Add labels and title
    ax.set_xlabel('X Coordinate (μm)', fontsize=12)
    ax.set_ylabel('Y Coordinate (μm)', fontsize=12)
    ax.set_title('LC Mesh with Filtered Data Points (Sagittal View)\n' +
                f'Filter: jem-status_reporter == "Positive" & injection region != "Non-Retro" & != "Thalamus"\n' +
                f'n = {len(df_filtered)} cells', fontsize=10)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    
    # Ensure equal aspect ratio is maintained
    ax.set_aspect('equal')
    
    # Adjust layout to prevent legend cutoff
    plt.tight_layout()
    
    # If saving is enabled, save the figure
    if if_save_fig:
        # Determine the output directory as the script's folder
        script_dir = os.path.dirname(os.path.abspath(__file__))

        # Save the figure as PNG
        output_filename = os.path.join(script_dir, 'fig_3a_ccf_projection.png')
        plt.savefig(output_filename, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        logger.info(f"Figure saved as: {output_filename}")

        # Also save as PDF for publication
        output_pdf = os.path.join(script_dir, 'fig_3a_ccf_projection.pdf')
        plt.savefig(output_pdf, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        logger.info(f"Figure saved as: {output_pdf}")
    else:
        # Show the plot
        plt.show()
    
    # Print summary statistics
    logger.info("\nSummary statistics:")
    logger.info(f"- Total filtered cells: {len(df_filtered)}")
    if len(df_filtered) > 0:
        logger.info(f"- X coordinate range: {df_filtered['x'].min():.1f} to {df_filtered['x'].max():.1f}")
        logger.info(f"- Y coordinate range: {df_filtered['y'].min():.1f} to {df_filtered['y'].max():.1f}")
        logger.info(f"- Injection regions: {', '.join(unique_regions)}")
    
    return fig, ax, df_filtered

if __name__ == "__main__":
    try:
        fig, ax, df_filtered = figure_3a_ccf_projection()
        print("Figure 3A generated successfully!")
    except Exception as e:
        logger.error(f"Error generating Figure 3A: {e}")
        raise
