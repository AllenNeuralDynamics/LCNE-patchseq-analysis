"""Init figures package"""

import matplotlib as mpl
import seaborn as sns

GLOBAL_FILTER = (
    "(`jem-status_reporter` == 'Positive') & "
    "(`injection region` != 'Non-Retro') & "
    "(`injection region` != 'Thalamus')"
)

GENE_FILTER = GLOBAL_FILTER + " & mapmycells_subclass_name.str.contains('DBH', case=False, na=False)"

# GLOBAL_FILTER += " & mapmycells_subclass_name.str.contains('DBH', case=False, na=False)"

# Global default area order for injection region plots
DEFAULT_AREA_ORDER = [
    "cortex",
    "spinal cord",
    "cerebellum",
]

def sort_region(region):
    """Sort injection regions based on DEFAULT_AREA_ORDER, with unknown regions at the end sorted alphabetically."""
    def _region_sort_key(region):
        region_lower = region.lower()
        if region_lower in DEFAULT_AREA_ORDER:
            return (DEFAULT_AREA_ORDER.index(region_lower), "")
        return (len(DEFAULT_AREA_ORDER), region_lower)
    return sorted(region, key=_region_sort_key)


def set_plot_style(base_size: int = 11, font_family: str = "Arial"):
    # Seaborn first (it may overwrite some rcParams)
    # sns.set_theme(context="paper", style="white", font_scale=1.0)
    mpl.rcParams.update({
        "font.family": font_family,
        "font.size": base_size,
        "axes.titlesize": base_size,
        "axes.labelsize": base_size,
        "xtick.labelsize": base_size - 1,
        "ytick.labelsize": base_size - 1,
        "legend.fontsize": base_size - 4,
        "legend.title_fontsize": base_size - 3,
        "figure.titlesize": base_size + 1,
        "pdf.fonttype": 42,        # editable text in Illustrator
        "ps.fonttype": 42,
    })