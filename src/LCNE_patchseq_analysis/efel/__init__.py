"""Extracting features using eFEL."""

import efel

from LCNE_patchseq_analysis import TIME_STEP

EFEL_SETTINGS = {
    "interp_step": TIME_STEP,
    "Threshold": 0.0,
}

# Set global eFEL settings
for setting, value in EFEL_SETTINGS.items():
    efel.api.set_setting(setting, value)
