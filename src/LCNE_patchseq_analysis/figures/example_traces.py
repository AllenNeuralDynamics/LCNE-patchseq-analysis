"""Raw example voltage traces for the three projection-target example cells (S14j).

Reads the raw peri-stimulus voltage sweeps straight out of the per-cell eFEL
feature .h5 files (no NWB access needed) and plots, for each example cell, a
suprathreshold spiking sweep on top and a near-rheobase sweep overlaid with the
most-hyperpolarized subthreshold sweep below — one column per projection target.
"""

import logging
import os

import numpy as np
import pandas as pd

from LCNE_patchseq_analysis import RAW_DIRECTORY, TIME_STEP
from LCNE_patchseq_analysis.figures import PROJECTION_COLORS

logger = logging.getLogger(__name__)

# Absolute path to the per-cell eFEL feature .h5 files, derived from the same
# data root as RAW_DIRECTORY so it stays correct across environments (e.g. in
# Code Ocean this resolves to /data/LCNE-patchseq-ephys/efel/features). Avoid a
# relative path, which breaks when the process is launched from a different CWD.
FEATURES_DIR = os.path.join(
    os.path.dirname(RAW_DIRECTORY), "efel", "features"
)

# Example cell per projection target, in display order. `region` keys into
# PROJECTION_COLORS so the trace colors match the S14k CDF panels.
EXAMPLE_CELLS = [
    {"label": "Isocortex", "region": "Cortex", "ephys_roi_id": "1388239233"},
    {"label": "Cerebellum", "region": "Cerebellum", "ephys_roi_id": "1426757704"},
    {"label": "Spinal cord", "region": "Spinal cord", "ephys_roi_id": "1410640556"},
]

# Vertical offset (mV) applied to the suprathreshold sweep so it sits above the
# rheobase / hyperpolarizing sweeps, mirroring the published layout.
SUPRA_OFFSET_MV = 120.0


def load_features(ephys_roi_id: str) -> dict:
    """Load the dict of DataFrames from a per-cell eFEL feature .h5 file."""
    path = os.path.join(FEATURES_DIR, f"{ephys_roi_id}_efel.h5")
    out = {}
    with pd.HDFStore(path, mode="r") as store:
        for key in store.keys():
            out[key.replace("/", "")] = store[key]
    return out


def _select_sweep(df, contains, stim_name, need_spikes, pick="min_abs_amp"):
    """Return one sweep_number matching the stimulus criteria, or None."""
    mask = (
        df["stimulus_code"].str.contains(contains)
        & (df["stimulus_name"] == stim_name)
    )
    if need_spikes:
        mask &= df["spike_count"] > 0
    sub = df[mask]
    if sub.empty:
        return None
    if pick == "min_abs_amp":
        row = sub.loc[sub["stimulus_amplitude"].abs().idxmin()]
    elif pick == "max_abs_amp":
        row = sub.loc[sub["stimulus_amplitude"].abs().idxmax()]
    else:
        raise ValueError(pick)
    return int(row["sweep_number"])


def select_sweeps(features: dict) -> dict:
    """Pick the suprathreshold, rheobase, and hyperpolarizing sweeps to plot."""
    df = features["df_features_per_sweep"].merge(
        features["df_sweeps"], on="sweep_number"
    )
    return {
        "supra": _select_sweep(df, "SupraThresh", "Long Square", True),
        "rheo": _select_sweep(df, "Rheo", "Long Square", True),
        "hyperpol": _select_sweep(df, "SubThresh", "Long Square", False, "max_abs_amp"),
    }


def plot_cell(ax, features: dict, sweeps: dict, color: str, linewidth: float = 1.0):
    """Plot the selected sweeps for one cell into ax."""
    raw = features["df_peri_stimulus_raw_traces"]

    def get_v(sweep_number):
        if sweep_number is None:
            return None, None
        row = raw.query("sweep_number == @sweep_number")
        if row.empty:
            return None, None
        v = np.asarray(row["V"].iloc[0], dtype=float)
        t = np.arange(len(v)) * TIME_STEP
        return t, v

    # Bottom: near-rheobase sweep overlaid with the hyperpolarizing sweep.
    for key in ("hyperpol", "rheo"):
        t, v = get_v(sweeps[key])
        if v is not None:
            ax.plot(t, v, color=color, lw=linewidth)

    # Top: suprathreshold spiking sweep, offset upward for clarity.
    t, v = get_v(sweeps["supra"])
    if v is not None:
        ax.plot(t, v + SUPRA_OFFSET_MV, color=color, lw=linewidth)

    ax.axis("off")


def add_scale_bar(ax, x_ms=200, y_mv=50, x0=0.02, y0=0.05):
    """Draw an L-shaped scale bar (time, voltage) in axes-fraction coords."""
    (x_lo, x_hi), (y_lo, y_hi) = ax.get_xlim(), ax.get_ylim()
    x_span, y_span = x_hi - x_lo, y_hi - y_lo
    xs = x_lo + x0 * x_span
    ys = y_lo + y0 * y_span
    ax.plot([xs, xs], [ys, ys + y_mv], color="black", lw=1.5)
    ax.plot([xs, xs + x_ms], [ys, ys], color="black", lw=1.5)
    ax.text(xs - 0.01 * x_span, ys + y_mv / 2, f"{y_mv} mV",
            ha="right", va="center", fontsize=9, rotation=90)
    ax.text(xs + x_ms / 2, ys - 0.02 * y_span, f"{x_ms} ms",
            ha="center", va="top", fontsize=9)


def plot_example_traces(axes, cells=EXAMPLE_CELLS, add_scalebar: bool = True):
    """Draw the example-trace column for each cell into the provided axes.

    Parameters
    ----------
    axes : sequence of matplotlib axes
        One axis per cell (same length as `cells`). Y-limits are shared so the
        scale bar on the first axis applies to all.
    cells : list of dict
        Each dict has 'label', 'region' (key into PROJECTION_COLORS), and
        'ephys_roi_id'.
    add_scalebar : bool
        Whether to draw an L-shaped scale bar on the first axis.
    """
    axes = np.atleast_1d(axes)

    # Raw traces live in FEATURES_DIR; if it's not mounted/available, skip the
    # raw-trace generation rather than crashing, leaving the axes blank.
    if not os.path.isdir(FEATURES_DIR):
        logger.warning(
            f"Features directory not found ({FEATURES_DIR}); "
            "skipping raw example-trace generation."
        )
        for ax in axes:
            ax.axis("off")
        return

    for ax, cell in zip(axes, cells):
        logger.info(f"Plotting {cell['label']} ({cell['ephys_roi_id']})...")
        features = load_features(cell["ephys_roi_id"])
        sweeps = select_sweeps(features)
        logger.info(f"  selected sweeps: {sweeps}")
        plot_cell(ax, features, sweeps, PROJECTION_COLORS[cell["region"]])
        ax.set_title(cell["label"], fontsize=13)

    # Share y-limits across all cells so one scale bar is valid for the row.
    y_lo = min(ax.get_ylim()[0] for ax in axes)
    y_hi = max(ax.get_ylim()[1] for ax in axes)
    for ax in axes:
        ax.set_ylim(y_lo, y_hi)

    if add_scalebar:
        add_scale_bar(axes[0])
