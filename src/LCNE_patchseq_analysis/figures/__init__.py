"""Init figures package"""

import matplotlib as mpl

GLOBAL_FILTER = (
    "(`jem-status_reporter` == 'Positive') & "
    "(`injection region` != 'Non-Retro') & "
    "(`injection region` != 'Thalamus')"
)

GENE_FILTER = (
    GLOBAL_FILTER + " & mapmycells_subclass_name.str.contains('DBH', case=False, na=False)"
)

# GLOBAL_FILTER += " & mapmycells_subclass_name.str.contains('DBH', case=False, na=False)"

# Global default area order for injection region plots
DEFAULT_AREA_ORDER = [
    "cortex",
    "spinal cord",
    "cerebellum",
]

# Default ephys feature mapping
DEFAULT_EPHYS_FEATURES = [
    # Manually selected ephys features to include in ANOVA analysis, multiple comparison correction,
    # and plots in supplementary figures.
    # Some of them should be renamed to improve readability in plots and to provide
    # explanations of their meanings.
    # Inclusion criteria:
    #     - Use ipfx pipeline for consistency with previous papers from the Allen Institute
    #     - The number of non-NaN values should be at least 100
    #        (without filtering, among all 333 cells)
    #     - Prefer long square rheobase, if relevant and available
    # Passive properties
    {"ipfx_tau": "tau"},
    {"ipfx_input_resistance_mohm_qc": "input_resistance"},
    {"ipfx_capacitance (pF)": "capacitance"},
    {"ipfx_v_baseline": "v_baseline"},
    {"ipfx_sag": "sag"},
    {"ipfx_sag_depol": "sag_depol"},  # ??
    {"ipfx_vm_for_sag": "vm_for_sag"},
    # Spike shape
    {"ipfx_threshold_v_rheo": "threshold_v_rheo"},
    {"ipfx_peak_deltav_rheo": "peak_deltav_rheo"},
    {"ipfx_peak_v_rheo": "peak_v_rheo"},
    {"ipfx_peak_v_adapt_ratio": "peak_v_adapt_ratio"},
    {"ipfx_postap_slope_rheo": "postap_slope_rheo"},
    {"ipfx_upstroke_rheo": "upstroke_rheo"},
    {"ipfx_downstroke_rheo": "downstroke_rheo"},
    {"ipfx_upstroke_downstroke_ratio_rheo": "upstroke_downstroke_ratio_rheo"},
    {"ipfx_ahp_delay_ratio_hero": "ahp_delay_ratio_hero"},
    {"ipfx_trough_v_rheo": "trough_v_rheo"},  # ??
    {"ipfx_trough_slowdeltav_rheo": "trough_slowdeltav_rheo"},  # ??
    {"ipfx_fast_trough_deltav_rheo": "fast_trough_deltav_rheo"},  # ??
    {"ipfx_fast_trough_v_rheo": "fast_trough_v_rheo"},  # ??
    {"ipfx_width_rheo": "width_rheo"},
    {"ipfx_width_adapt_ratio": "width_adapt_ratio"},
    # Spike train features
    {"ipfx_rheobase_i": "rheobase_i"},
    {"ipfx_fi_fit_slope": "fi_fit_slope"},
    {"ipfx_avg_rate_max": "avg_rate_max"},
    {"ipfx_latency_rheo": "latency_rheo"},
    {"ipfx_isi_cv_mean": "isi_cv_mean"},
    {"ipfx_isi_adapt_ratio": "isi_adapt_ratio"},
    {"ipfx_adapt_hero": "adapt_hero"},
    {"ipfx_first_isi_inv_hero": "first_isi_inv_hero"},
]


def sort_region(region):
    """Sort injection regions based on DEFAULT_AREA_ORDER, with unknown
    regions at the end sorted alphabetically."""

    def _region_sort_key(region):
        region_lower = region.lower()
        if region_lower in DEFAULT_AREA_ORDER:
            return (DEFAULT_AREA_ORDER.index(region_lower), "")
        return (len(DEFAULT_AREA_ORDER), region_lower)

    return sorted(region, key=_region_sort_key)


def set_plot_style(base_size: int = 11, font_family: str = "Arial"):
    # Seaborn first (it may overwrite some rcParams)
    # sns.set_theme(context="paper", style="white", font_scale=1.0)
    mpl.rcParams.update(
        {
            "font.family": font_family,
            "font.size": base_size,
            "axes.titlesize": base_size,
            "axes.labelsize": base_size,
            "xtick.labelsize": base_size - 1,
            "ytick.labelsize": base_size - 1,
            "legend.fontsize": base_size - 4,
            "legend.title_fontsize": base_size - 3,
            "figure.titlesize": base_size + 1,
            "pdf.fonttype": 42,  # editable text in Illustrator
            "ps.fonttype": 42,
        }
    )
