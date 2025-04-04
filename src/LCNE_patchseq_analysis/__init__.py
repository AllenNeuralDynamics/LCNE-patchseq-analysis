"""Init package"""

import logging
import os

logger = logging.getLogger(__name__)


__version__ = "0.10.0"

# Get the path of this file
PACKAGE_DIRECTORY = os.path.dirname(os.path.abspath(__file__))
RAW_DIRECTORY = os.path.join(PACKAGE_DIRECTORY, "../../data/LCNE-patchseq-ephys/raw/")
RESULTS_DIRECTORY = os.path.join(PACKAGE_DIRECTORY, "../../results/")
TIME_STEP = 0.02  # ms

# Check if Raw data directory exists
if not os.path.exists(RAW_DIRECTORY):
    logger.warning(f"Raw data directory does not exist: {RAW_DIRECTORY}")
