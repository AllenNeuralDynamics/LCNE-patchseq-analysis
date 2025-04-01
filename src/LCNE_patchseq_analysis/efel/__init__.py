"""Extracting features using eFEL."""

import efel

EFEL_SETTINGS = {
    "interp_step": 0.02,
    "Threshold": 0.0,
}

# Set global eFEL settings
for setting, value in EFEL_SETTINGS.items():
    efel.api.set_setting(setting, value)
