"""Init figures package"""

# Global filter for patch-seq analysis figures
GLOBAL_FILTER = (
    "(`jem-status_reporter` == 'Positive') & "
    "(`injection region` != 'Non-Retro') & "
    "(`injection region` != 'Thalamus')"
)

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
