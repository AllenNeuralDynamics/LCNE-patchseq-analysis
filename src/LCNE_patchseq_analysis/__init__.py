"""Init package"""

import logging
import os

logger = logging.getLogger(__name__)


__version__ = "0.11.0"

# Get the path of this file
PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RAW_DIRECTORY = os.path.join(PACKAGE_DIRECTORY, "../../data/LCNE-patchseq-ephys/raw/")
RESULTS_DIRECTORY = os.path.join(PACKAGE_DIRECTORY, "../../results/")
TIME_STEP = 0.02  # ms

REGION_COLOR_MAPPER = {
    "C5": "pink",
    "PL": "green",
    "PL, MOs": "green",
    "Cortex": "green",
    "VAL": "red",
    "Crus 1": "gold",
    "Non-Retro": "grey",
}

# Add lowercase versions of all region keys to the color mapper
lowercase_regions = {}
for region, color in list(REGION_COLOR_MAPPER.items()):
    lowercase_region = region.lower()
    if lowercase_region not in REGION_COLOR_MAPPER:
        lowercase_regions[lowercase_region] = color

# Update the color mapper with lowercase versions
REGION_COLOR_MAPPER.update(lowercase_regions)


# Check if Raw data directory exists
if not os.path.exists(RAW_DIRECTORY):
    logger.warning(f"Raw data directory does not exist: {RAW_DIRECTORY}")
