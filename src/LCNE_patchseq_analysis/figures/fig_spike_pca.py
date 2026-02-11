"""Spike waveform PCA analysis and figure generation.

Reproduces the "Spike Analysis" panel from LCNE-patchseq-viz as static
matplotlib figures, using the same default settings.

Interactive version:
https://hanhou-patchseq.hf.space/patchseq_panel_viz?tab=1
"""

import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.decomposition import PCA

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.data_util.mesh import plot_mesh
from LCNE_patchseq_analysis.figures import set_plot_style
from LCNE_patchseq_analysis.figures.util import save_figure
from LCNE_patchseq_analysis.pipeline_util.s3 import (
    get_public_representative_spikes,
    load_mesh_from_s3,
)
from LCNE_patchseq_analysis.population_analysis.spikes import (
    extract_representative_spikes,
)

logger = logging.getLogger(__name__)

# --- Default parameters matching the viz app widget defaults ---
DEFAULT_SPIKE_TYPE = "average"
DEFAULT_EXTRACT_FROM = "long_square_rheo, min"
DEFAULT_SPIKE_RANGE = (-3, 6)  # ms
DEFAULT_NORMALIZE_WINDOW_V = (-2, 4)  # ms
DEFAULT_NORMALIZE_WINDOW_DVDT = (-2, 0)  # ms

SPINAL_REGIONS = ["C5", "Spinal cord"]
CORTEX_REGIONS = ["Cortex", "PL", "PL, MOs"]
CB_REGIONS = ["Crus 1", "Cerebellum", "CB"]


def spike_pca_analysis(
    df_meta: pd.DataFrame,
    df_spikes: pd.DataFrame | None = None,
    spike_type: str = DEFAULT_SPIKE_TYPE,
    extract_from: str = DEFAULT_EXTRACT_FROM,
    spike_range: tuple = DEFAULT_SPIKE_RANGE,
    normalize_window_v: tuple = DEFAULT_NORMALIZE_WINDOW_V,
    normalize_window_dvdt: tuple = DEFAULT_NORMALIZE_WINDOW_DVDT,
    filtered_df_meta: pd.DataFrame | None = None,
) -> dict:
    """Run PCA on spike waveforms.

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata with columns including 'ephys_roi_id', 'injection region',
        'X (A --> P)', 'Y (D --> V)', and optionally ipfx_tau columns.
    df_spikes : pd.DataFrame, optional
        Pre-loaded spike waveforms. If None, loads from S3 using spike_type.
    spike_type : str
        Which spike to use: "average", "first", "second", or "last".
    extract_from : str
        Stimulus source to extract spikes from.
    spike_range : tuple
        (start, end) in ms — time window of spike waveform to analyse.
    normalize_window_v : tuple
        (start, end) in ms — window for V min-max normalization.
    normalize_window_dvdt : tuple
        (start, end) in ms — window for dV/dt min-max normalization.
    filtered_df_meta : pd.DataFrame, optional
        If provided, restrict analysis to these cells.

    Returns
    -------
    dict with keys: df_v_proj, df_v_norm, reducer, tau_col
    """
    if df_spikes is None:
        df_spikes = get_public_representative_spikes(spike_type)

    # Extract normalised, peak-aligned waveforms
    df_v_norm, _df_dvdt_norm = extract_representative_spikes(
        df_spikes=df_spikes,
        extract_from=extract_from,
        if_normalize_v=True,
        normalize_window_v=normalize_window_v,
        if_normalize_dvdt=True,
        normalize_window_dvdt=normalize_window_dvdt,
        if_smooth_dvdt=False,
        if_align_dvdt_peaks=True,
        filtered_df_meta=filtered_df_meta if filtered_df_meta is not None else df_meta,
    )

    # Filter to spike_range (cast columns to numeric for static type checkers)
    time_cols = np.asarray(
        pd.to_numeric(df_v_norm.columns, errors="coerce"), dtype=float
    )
    in_window = (
        np.isfinite(time_cols)
        & (time_cols >= spike_range[0])
        & (time_cols <= spike_range[1])
    )
    df_v_norm = df_v_norm.loc[:, in_window]

    # PCA
    v = df_v_norm.values
    reducer = PCA()
    v_proj = reducer.fit_transform(v)
    n_components = 5
    columns = [f"PCA{i}" for i in range(1, n_components + 1)]
    df_v_proj = pd.DataFrame(v_proj[:, :n_components], index=df_v_norm.index)
    df_v_proj.columns = columns

    # Merge metadata
    tau_col = [c for c in df_meta.columns if "ipfx_tau" in c][0]
    x_col, y_col = ("X (A --> P)", "Y (D --> V)")
    if x_col not in df_meta.columns:
        x_col, y_col = ("x", "y")

    merge_cols = ["ephys_roi_id", "injection region", x_col, y_col, tau_col]
    df_v_proj = df_v_proj.merge(df_meta[merge_cols], on="ephys_roi_id", how="left")

    if x_col == "x":
        df_v_proj = df_v_proj.rename(columns={"x": "X (A --> P)", "y": "Y (D --> V)"})

    # ipfx_tau is in seconds; convert to milliseconds for plotting.
    tau_ms_col = "membrane_time_constant_ms"
    df_v_proj[tau_ms_col] = pd.to_numeric(df_v_proj[tau_col], errors="coerce") * 1000

    return {
        "df_v_proj": df_v_proj,
        "df_v_norm": df_v_norm,
        "reducer": reducer,
        "tau_col": tau_ms_col,
    }


# ---------------------------------------------------------------------------
# Private plotting helpers
# ---------------------------------------------------------------------------


def _plot_pca_scatter(ax, df_v_proj, marker_size=40):
    """PC1 vs PC2 scatter coloured by projection target."""
    for region in df_v_proj["injection region"].unique():
        sub = df_v_proj.query("`injection region` == @region")
        color = REGION_COLOR_MAPPER.get(region, "gray")
        ax.scatter(
            sub["PCA1"],
            sub["PCA2"],
            c=color,
            s=marker_size,
            alpha=0.6,
            label=f"{region} (n={len(sub)})",
            edgecolors="none",
        )

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_aspect("equal")
    ax.legend(fontsize=6, loc="best", framealpha=0.5)
    sns.despine(ax=ax, trim=True)


def _plot_box_strip(ax, groups, marker_size=15, alpha=0.5, seed=42):
    """Jittered strip plot with violin overlay.

    Parameters
    ----------
    groups : list of (label, data_array, color) tuples
    """
    rng = np.random.default_rng(seed)
    for idx, (_, data, color) in enumerate(groups):
        # Violin body
        vp = ax.violinplot(
            dataset=[data],
            positions=[idx],
            widths=0.7,
            showmeans=False,
            showmedians=False,
            showextrema=False,
        )
        for body in vp["bodies"]:
            body.set_facecolor(color)
            body.set_edgecolor("black")
            body.set_alpha(0.25)

        # Median marker
        median = np.median(data)
        ax.hlines(median, idx - 0.2, idx + 0.2, color="black", linewidth=2, zorder=4)

        # Jittered observations on top
        jitter = rng.uniform(-0.15, 0.15, len(data))
        ax.scatter(
            np.full(len(data), idx) + jitter,
            data,
            s=marker_size,
            c=color,
            alpha=alpha,
            edgecolors="black",
            linewidths=0.3,
            zorder=3,
        )

    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([f"{lbl}\n(n={len(d)})" for lbl, d, _ in groups])
    ax.set_xlim(-0.6, len(groups) - 0.4)
    sns.despine(ax=ax, trim=True)


def _plot_spatial_map(ax, df_v_proj, color_col, cmap, label, marker_size=40):
    """Scatter on LC mesh (sagittal view) coloured by a continuous variable."""
    mesh = load_mesh_from_s3()
    plot_mesh(ax, mesh, direction="sagittal", meshcol="lightgray")

    vals = pd.to_numeric(df_v_proj[color_col], errors="coerce")

    sc = ax.scatter(
        df_v_proj["X (A --> P)"],
        df_v_proj["Y (D --> V)"],
        c=vals,
        cmap=cmap,
        s=marker_size,
        edgecolors=[
            REGION_COLOR_MAPPER.get(r, "black") for r in df_v_proj["injection region"]
        ],
        linewidths=1,
        alpha=0.7,
    )
    plt.colorbar(sc, ax=ax, shrink=0.7, label=label)
    ax.set_xlabel("Anterior-posterior (μm)")
    ax.set_ylabel("Dorsal-ventral (μm)")
    ax.set_aspect("equal")
    sns.despine(ax=ax, trim=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def figure_spike_pca(
    df_meta: pd.DataFrame,
    df_spikes: pd.DataFrame | None = None,
    spike_type: str = DEFAULT_SPIKE_TYPE,
    extract_from: str = DEFAULT_EXTRACT_FROM,
    spike_range: tuple = DEFAULT_SPIKE_RANGE,
    normalize_window_v: tuple = DEFAULT_NORMALIZE_WINDOW_V,
    normalize_window_dvdt: tuple = DEFAULT_NORMALIZE_WINDOW_DVDT,
    filtered_df_meta: pd.DataFrame | None = None,
    if_save_figure: bool = True,
    figsize: tuple = (18, 10),
):
    """Generate the spike PCA figure (2 rows, 5 panels).

    Top row: PCA scatter | PC1 violin plot | membrane time constant violin plot
    Bottom row: PC1 spatial | membrane time constant spatial

    Parameters
    ----------
    df_meta : pd.DataFrame
        Metadata (from load_ephys_metadata).
    df_spikes : pd.DataFrame, optional
        Pre-loaded spike waveforms. Loaded from S3 if None.
    spike_type, extract_from, spike_range, normalize_window_v,
    normalize_window_dvdt :
        Analysis parameters (see spike_pca_analysis).
    filtered_df_meta : pd.DataFrame, optional
        Cell-level filter.
    if_save_figure : bool
        Whether to save figure to results/figures/.
    figsize : tuple
        Figure size in inches.

    Returns
    -------
    (fig, axes_dict, results)
    """
    set_plot_style(base_size=12, font_family="Helvetica")

    results = spike_pca_analysis(
        df_meta=df_meta,
        df_spikes=df_spikes,
        spike_type=spike_type,
        extract_from=extract_from,
        spike_range=spike_range,
        normalize_window_v=normalize_window_v,
        normalize_window_dvdt=normalize_window_dvdt,
        filtered_df_meta=filtered_df_meta,
    )
    df_v_proj = results["df_v_proj"]
    tau_col = results["tau_col"]

    # --- Build figure ---
    fig = plt.figure(figsize=figsize)
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.35)
    gs_top = gs[0].subgridspec(1, 3, width_ratios=[3, 1, 1], wspace=0.35)
    gs_bottom = gs[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.3)
    axes = [
        fig.add_subplot(gs_top[0, 0]),
        fig.add_subplot(gs_top[0, 1]),
        fig.add_subplot(gs_top[0, 2]),
        fig.add_subplot(gs_bottom[0, 0]),
        fig.add_subplot(gs_bottom[0, 1]),
    ]
    axes_dict = {}

    # Panel 1: PCA scatter
    ax = axes[0]
    axes_dict["pca_scatter"] = ax
    _plot_pca_scatter(ax, df_v_proj)
    ax.set_title("PC1 vs PC2")

    # Panel 2: PC1 violin plot by projection target
    ax = axes[1]
    axes_dict["pca1_box"] = ax
    pc1_groups = _build_projection_groups(df_v_proj, "PCA1")
    _plot_box_strip(ax, pc1_groups)
    ax.set_title("PC1 (norm. V)")
    ax.set_ylabel("PC1")

    # Panel 3: membrane time constant violin plot by projection target
    ax = axes[2]
    axes_dict["tau_box"] = ax
    tau_groups = _build_projection_groups(df_v_proj, tau_col)
    _plot_box_strip(ax, tau_groups)
    ax.set_title("Membrane time constant (ms)")
    ax.set_ylabel("Membrane time constant (ms)")

    # Panel 4: PC1 in X/Y space with LC mesh
    ax = axes[3]
    axes_dict["pca1_spatial"] = ax
    _plot_spatial_map(ax, df_v_proj, "PCA1", "RdBu_r", "PC1")
    ax.set_title("PC1 in CCF space")

    # Panel 5: membrane time constant in X/Y space with LC mesh
    ax = axes[4]
    axes_dict["tau_spatial"] = ax
    _plot_spatial_map(ax, df_v_proj, tau_col, "inferno", "Membrane time constant (ms)")
    ax.set_title("Membrane time constant (ms) in CCF space")

    fig.tight_layout()

    if if_save_figure:
        save_figure(
            fig, filename="fig_spike_pca", formats=("png", "svg"), bbox_inches="tight"
        )

    return fig, axes_dict, results


def _build_projection_groups(df_v_proj, value_col):
    """Build (label, values, color) groups for spinal cord, cortex, and CB."""
    groups = []
    for grp_label, region_set, color in [
        ("Spinal cord", SPINAL_REGIONS, REGION_COLOR_MAPPER["Spinal cord"]),
        ("Cortex", CORTEX_REGIONS, REGION_COLOR_MAPPER["Cortex"]),
        ("CB", CB_REGIONS, REGION_COLOR_MAPPER["Cerebellum"]),
    ]:
        mask = df_v_proj["injection region"].isin(region_set)
        vals_series = pd.Series(
            pd.to_numeric(df_v_proj.loc[mask, value_col], errors="coerce")
        )
        vals = vals_series.dropna().to_numpy()
        groups.append((grp_label, vals, color))
    return groups


if __name__ == "__main__":
    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
    from LCNE_patchseq_analysis.figures import GLOBAL_FILTER

    logging.basicConfig(level=logging.INFO)
    logger.info("Loading metadata...")
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    df_meta = df_meta.query(GLOBAL_FILTER)
    logger.info(f"Loaded metadata with shape: {df_meta.shape}")

    logger.info("Loading spike waveforms...")
    df_spikes = get_public_representative_spikes("average")

    logger.info("Generating spike PCA figure...")
    fig, axes_dict, results = figure_spike_pca(
        df_meta, df_spikes, filtered_df_meta=df_meta
    )
