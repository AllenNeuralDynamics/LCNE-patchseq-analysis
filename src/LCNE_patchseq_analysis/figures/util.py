import os
from matplotlib.figure import Figure
import logging

logger = logging.getLogger()

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
