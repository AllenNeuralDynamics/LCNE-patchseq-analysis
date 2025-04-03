"""Extracting features using eFEL."""

import efel
import json
import os
from LCNE_patchseq_analysis import TIME_STEP

EFEL_SETTINGS = {
    "interp_step": TIME_STEP,
    "Threshold": -10.0,
    "strict_stiminterval": False,
}

# Set global eFEL settings
for setting, value in EFEL_SETTINGS.items():
    efel.api.set_setting(setting, value)

# Load non-scalar features
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "efel_per_spike_features.json"), "r") as f:
    EFEL_PER_SPIKE_FEATURES = json.load(f)
