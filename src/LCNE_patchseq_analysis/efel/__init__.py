"""Extracting features using eFEL."""

import json
import os

import efel

from LCNE_patchseq_analysis import TIME_STEP

# ---- Global eFEL settings ---
EFEL_SETTINGS = {
    "interp_step": TIME_STEP,
    "Threshold": -10.0,
    "strict_stiminterval": False,
}
for setting, value in EFEL_SETTINGS.items():
    efel.api.set_setting(setting, value)

# Load non-scalar features
# Get the directory of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))
with open(os.path.join(current_dir, "efel_per_spike_features.json"), "r") as f:
    EFEL_PER_SPIKE_FEATURES = json.load(f)


# ---- Cell-level eFEL features ---
# To be explicit so that we can see the available options clearly.
# Feature columns will be all combinations of the "Features" and "Froms" below,
# for "spike" and "sag" respectively.
#
# For the "Features":
#     - See eFEL documentation for details. (https://efel.readthedocs.io/en/latest/eFeatures.html)
#     - "first_spike_" prefix means all features are extracted from the first spike.
#     - "sag_" prefix means all features are extracted from the sag.
#
# For the "Froms":
#     - The first element is the stimulus type.
#         - "rheo" means rheobase in either short or long square
#         - "supra" means supra-threshold (in long square only)
#     - The second element is the aggregation method.
#         - "min" means the minimum amplitude that evokes at least one spike
#         - "aver" means the average amplitude of all sweeps in one "from"

EXTRACT_SPIKE_FEATURES = [
    "first_spike_ADP_peak_amplitude",
    "first_spike_ADP_peak_values",
    "first_spike_AHP_depth",
    "first_spike_AHP_depth_abs",
    "first_spike_AHP_depth_from_peak",
    "first_spike_AHP_time_from_peak",
    "first_spike_AP_amplitude",
    "first_spike_AP_amplitude_from_voltagebase",
    "first_spike_AP_begin_voltage",
    "first_spike_AP_begin_width",
    "first_spike_AP_duration",
    "first_spike_AP_duration_half_width",
    "first_spike_AP_fall_rate",
    "first_spike_AP_fall_time",
    "first_spike_AP_height",
    "first_spike_AP_peak_downstroke",
    "first_spike_AP_peak_upstroke",
    "first_spike_AP_phaseslope",
    "first_spike_AP_rise_rate",
    "first_spike_AP_rise_time",
    "first_spike_AP_width",
    "first_spike_AP_width_between_threshold",
    "first_spike_min_AHP_values",
    "first_spike_min_between_peaks_values",
    "first_spike_peak_voltage",
    "first_spike_spike_half_width",
    "efel_first_spike_AP_width",
]

EXTRACT_SPIKE_FROMS = {
    # name: [stimulus_type, aggregation_method]
    "short_square_rheo, min": ["short_square_rheo", "min"],
    "short_square_rheo, aver": ["short_square_rheo", "aver"],
    "long_square_rheo, min": ["long_square_rheo", "min"],
    "long_square_rheo, aver": ["long_square_rheo", "aver"],
    "long_square_supra, min": ["long_square_supra", "min"],
    "long_square_supra, aver": ["long_square_supra", "aver"],
}

EXTRACT_SAG_FEATURES = ["sag_amplitude", "sag_ratio1", "sag_ratio2", "sag_time_constant"]

EXTRACT_SAG_FROMS = {
    # name: [stimulus_type, threshold_amplitude]
    "subthreshold, 50": ["subthreshold", 50],
    "subthreshold, 90": ["subthreshold", 90],
    "subthreshold, aver": ["subthreshold", "aver"],
}

# ---- Cell-level summary plots ----
# Control which sweeps to show in the cell-level summary plots.
CELL_SUMMARY_PLOT_SHOW_SWEEPS = [
    {"label": "supra, min", "stim_type": ["long_square_supra", "min"], "color": "salmon"},
    {"label": "rheo, min", "stim_type": ["long_square_rheo", "min"], "color": "deepskyblue"},
    {"label": "-50 pA", "stim_type": ["subthreshold", 50], "color": "gray"},
    {"label": "-90 pA", "stim_type": ["subthreshold", 90], "color": "black"},
]

# Control which spikes to overlay in the cell-level summary plots. (only the first spike)
CELL_SUMMARY_PLOT_SHOW_SPIKES = [
    {
        "label": "short_square_rheo, min",
        "stim_type": ["short_square_rheo", "min"],
        "color": "darkgreen",
    },
    {
        "label": "long_square_supra, min",
        "stim_type": ["long_square_supra", "min"],
        "color": "salmon",
    },
    {
        "label": "long_square_rheo, min",
        "stim_type": ["long_square_rheo", "min"],
        "color": "deepskyblue",
    },
]
