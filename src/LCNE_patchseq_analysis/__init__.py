"""Init package"""

import logging
import os

logger = logging.getLogger(__name__)


__version__ = "0.34.6"

# Get the path of this file
PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))

IN_CODEOCEAN = bool(os.getenv("CO_COMPUTATION_ID"))

if IN_CODEOCEAN:
    RAW_DIRECTORY = "/data/LCNE-patchseq-ephys/raw"
    MORPHOLOGY_DIRECTORY = "/data/LCNE-patchseq-ephys/morphology"
    RESULTS_DIRECTORY = "/results"
elif os.name == "posix":  # Mac/Linux
    RAW_DIRECTORY = "/Users/han.hou/Scripts/LCNE-patchseq-analysis/data/LCNE-patchseq-ephys/raw"
    MORPHOLOGY_DIRECTORY = (
        "/Users/han.hou/Scripts/LCNE-patchseq-analysis/data/LCNE-patchseq-ephys/morphology"
    )
    RESULTS_DIRECTORY = os.path.join(PACKAGE_DIRECTORY, "../../results/")
else:  # Windows
    RAW_DIRECTORY = os.path.join(R"E:\s3\aind-patchseq-data\raw")
    MORPHOLOGY_DIRECTORY = os.path.join(R"E:\s3\aind-patchseq-data\morphology")
    RESULTS_DIRECTORY = os.path.join(PACKAGE_DIRECTORY, "../../results/")

# Mode announcement (INFO level so it appears at normal logging)
if IN_CODEOCEAN:
    logger.info("=" * 60)
    logger.info("Running in CodeOcean mode")
    logger.info(f"  RAW data  : {RAW_DIRECTORY}")
    logger.info(f"  Results   : {RESULTS_DIRECTORY}")
    logger.info("  Metadata  : results folder if available, else S3")
    logger.info("=" * 60)
else:
    logger.info("Running in S3/local mode")
    logger.info(f"  RAW data  : {RAW_DIRECTORY}")
    logger.info(f"  Results   : {RESULTS_DIRECTORY}")
TIME_STEP = 0.02  # ms

REGION_COLOR_MAPPER = {
    "C5": "pink",
    "Spinal cord": "pink",
    "Cortex": "green",
    "PL": "green",
    "PL, MOs": "green",
    "VAL": "red",
    "Thalamus": "red",
    "Crus 1": "gold",
    "Cerebellum": "gold",
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
