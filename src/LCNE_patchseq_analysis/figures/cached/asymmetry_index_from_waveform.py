import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER, RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.mesh import plot_mesh
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures import (
    DEFAULT_EPHYS_FEATURES,
    GLOBAL_FILTER,
    set_plot_style,
)
from LCNE_patchseq_analysis.figures.util import generate_violin_plot, save_figure
from LCNE_patchseq_analysis.figures.cached.fig_3c import (
    _generate_multi_feature_scatter_plots,
)
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes
from LCNE_patchseq_analysis.pipeline_util.s3 import load_mesh_from_s3
from LCNE_patchseq_analysis.population_analysis.anova import anova_features

logger = logging.getLogger(__name__)


def find_local_maxima(values: np.ndarray) -> np.ndarray:
    """Return indices of local maxima in a 1D array."""
    if values.size < 3:
        return np.array([], dtype=int)
    return np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1


def compute_spike_metrics(times_ms: np.ndarray, voltages: np.ndarray) -> dict:
    """Compute dv/dt-based timing metrics for a single spike waveform."""
    # Compute rise/fall timing from dv/dt features.
    dv = np.gradient(voltages, times_ms)
    peak_deriv_idx = int(np.argmax(dv))
    post_peak_slice = dv[peak_deriv_idx + 1 :]
    trough_deriv_idx = (
        int(peak_deriv_idx + 1 + np.argmin(post_peak_slice))
        if post_peak_slice.size > 0
        else None
    )

    fall_time_ms = (
        times_ms[trough_deriv_idx] - times_ms[peak_deriv_idx]
        if trough_deriv_idx is not None
        else np.nan
    )

    peak_idx = int(np.argmax(voltages))
    post_peak_voltage = voltages[peak_idx + 1 :]
    trough_idx = (
        int(peak_idx + 1 + np.argmin(post_peak_voltage))
        if post_peak_voltage.size > 0
        else None
    )

    kick_idx = None
    baseline_mask = times_ms <= (times_ms.min() + 1.0)
    if np.any(baseline_mask):
        baseline_dv = dv[baseline_mask]
        baseline_mean = float(np.nanmean(baseline_dv))
        baseline_std = float(np.nanstd(baseline_dv))
    else:
        baseline_mean = float(np.nanmean(dv))
        baseline_std = float(np.nanstd(dv))

    peak_dv = dv[peak_deriv_idx]
    dv_threshold = max(baseline_mean + 3.0 * baseline_std, 0.1 * peak_dv)
    local_maxima = find_local_maxima(dv[:peak_deriv_idx])
    if local_maxima.size > 0:
        candidates = [idx for idx in local_maxima if dv[idx] >= dv_threshold]
        if candidates:
            kick_idx = int(candidates[0])
        else:
            kick_idx = int(local_maxima[-1])

    rise_time_ms = (
        times_ms[peak_deriv_idx] - times_ms[kick_idx]
        if kick_idx is not None
        else np.nan
    )

    time_asymmetry = (
        rise_time_ms / fall_time_ms
        if np.isfinite(fall_time_ms) and fall_time_ms != 0
        else np.nan
    )

    return {
        "kick_idx": kick_idx,
        "peak_deriv_idx": peak_deriv_idx,
        "trough_deriv_idx": trough_deriv_idx,
        "rise_time_ms": rise_time_ms,
        "fall_time_ms": fall_time_ms,
        "time_asymmetry": time_asymmetry,
        "peak_idx": peak_idx,
        "trough_idx": trough_idx,
        "dv": dv,
    }


def compute_metrics_for_table(df_spikes: pd.DataFrame, spike_type: str):
    """Compute timing metrics and keep waveform vectors for PCA."""
    df_spikes = df_spikes.copy()
    df_spikes.index = df_spikes.index.set_levels(
        df_spikes.index.levels[0].astype(str), level=0
    )
    times_ms = df_spikes.columns.to_numpy(dtype=float)
    records = []
    waveforms = []
    derivatives = []

    for (ephys_roi_id, extract_from), row in df_spikes.iterrows():
        voltages = row.to_numpy(dtype=float)
        metrics = compute_spike_metrics(times_ms, voltages)
        waveforms.append(voltages)
        derivatives.append(metrics.pop("dv"))
        records.append(
            {
                "ephys_roi_id": str(ephys_roi_id),
                "extract_from": extract_from,
                "spike_type": spike_type,
                **metrics,
            }
        )

    df_metrics = pd.DataFrame.from_records(records)
    df_metrics["waveform_vector"] = list(waveforms)
    df_metrics["dvdt_vector"] = list(derivatives)
    return df_metrics, times_ms


def drop_subthreshold(df_spikes: pd.DataFrame) -> pd.DataFrame:
    """Remove subthreshold stimulus rows based on extract_from label."""
    if not isinstance(df_spikes.index, pd.MultiIndex):
        return df_spikes
    extract_from = df_spikes.index.get_level_values(1).astype(str)
    return df_spikes.loc[~extract_from.str.contains("subthreshold", case=False)]


def compute_pc1(features: np.ndarray) -> np.ndarray:
    """Compute a whitened PC1 score for each row in a feature matrix."""
    # Standardize features then compute whitened PC1.
    valid_mask = np.isfinite(features).all(axis=1)
    pc1 = np.full(features.shape[0], np.nan)
    if valid_mask.sum() > 2:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features[valid_mask])
        pca = PCA(n_components=1, whiten=True)
        pc1_vals = pca.fit_transform(features_scaled).ravel()
        pc1[valid_mask] = pc1_vals
    return pc1


def add_pca_variants(df_metrics: pd.DataFrame, ephys_cols: list[str]) -> pd.DataFrame:
    """Add multiple PCA projections to the metrics table."""
    # Generate PCA variants across spike types.
    df_metrics = df_metrics.copy()
    for spike_type, df_group in df_metrics.groupby("spike_type"):
        idx = df_group.index
        waveforms = np.vstack(df_group["waveform_vector"].to_numpy())
        derivatives = np.vstack(df_group["dvdt_vector"].to_numpy())
        derived = df_group[["rise_time_ms", "fall_time_ms", "time_asymmetry"]].to_numpy(
            dtype=float
        )
        if ephys_cols:
            ephys_values = (
                df_group[ephys_cols]
                .apply(pd.to_numeric, errors="coerce")
                .to_numpy(dtype=float)
            )
        else:
            ephys_values = None
        df_metrics.loc[idx, "pc1_waveform"] = compute_pc1(waveforms)
        df_metrics.loc[idx, "pc1_dvdt"] = compute_pc1(derivatives)

        derived_ephys = (
            np.column_stack([derived, ephys_values])
            if ephys_values is not None
            else derived
        )
        df_metrics.loc[idx, "pc1_derived_ephys"] = compute_pc1(derived_ephys)

        all_features = (
            np.column_stack([waveforms, derivatives, derived, ephys_values])
            if ephys_values is not None
            else np.column_stack([waveforms, derivatives, derived])
        )
        df_metrics.loc[idx, "pc1_all"] = compute_pc1(all_features)

    return df_metrics


def plot_example_spike(
    df_spikes: pd.DataFrame,
    df_metrics: pd.DataFrame,
    times_ms: np.ndarray,
    ephys_roi_id: str,
    extract_from: str,
    spike_type: str,
    output_dir: str,
):
    """Plot waveform and dv/dt diagnostics for a single example."""
    key = (str(ephys_roi_id), extract_from)
    if key not in df_spikes.index:
        return
    voltage = df_spikes.loc[key].to_numpy(dtype=float)
    metrics = df_metrics.query(
        "ephys_roi_id == @ephys_roi_id and extract_from == @extract_from "
        "and spike_type == @spike_type"
    )
    if metrics.empty:
        return
    metrics = metrics.iloc[0]
    dv = np.gradient(voltage, times_ms)

    fig, axes = plt.subplots(2, 1, figsize=(6, 6), sharex=True)
    ax_wave, ax_dv = axes
    ax_wave.plot(times_ms, voltage, color="black", linewidth=1.5)
    ax_wave.scatter(
        times_ms[int(metrics.peak_idx)],
        voltage[int(metrics.peak_idx)],
        color="red",
        label="peak",
        zorder=5,
    )
    if pd.notnull(metrics.trough_idx):
        ax_wave.scatter(
            times_ms[int(metrics.trough_idx)],
            voltage[int(metrics.trough_idx)],
            color="blue",
            label="trough",
            zorder=5,
        )
    ax_wave.set_ylabel("Voltage (mV)")
    ax_wave.legend(frameon=False)
    ax_wave.set_title(f"{ephys_roi_id} | {extract_from} | {spike_type}")

    ax_dv.plot(times_ms, dv, color="purple", linewidth=1.2)
    if pd.notnull(metrics.kick_idx):
        ax_dv.scatter(
            times_ms[int(metrics.kick_idx)],
            dv[int(metrics.kick_idx)],
            color="green",
            label="start?",
            zorder=5,
        )
    ax_dv.scatter(
        times_ms[int(metrics.peak_deriv_idx)],
        dv[int(metrics.peak_deriv_idx)],
        color="red",
        label="dV/dt peak",
        zorder=5,
    )
    if pd.notnull(metrics.trough_deriv_idx):
        ax_dv.scatter(
            times_ms[int(metrics.trough_deriv_idx)],
            dv[int(metrics.trough_deriv_idx)],
            color="blue",
            label="dV/dt trough",
            zorder=5,
        )
    ax_dv.set_xlabel("Time to peak (ms)")
    ax_dv.set_ylabel("dV/dt")
    ax_dv.legend(frameon=False)

    os.makedirs(output_dir, exist_ok=True)
    save_figure(
        fig,
        output_dir=output_dir,
        filename=f"asymmetry_example_{ephys_roi_id}_{extract_from}_{spike_type}",
        formats=("png",),
        bbox_inches="tight",
    )
    plt.close(fig)


def sanitize_filename(text: str) -> str:
    """Normalize text for file-friendly names."""
    return text.replace(",", "").replace(" ", "_").replace("/", "-").replace("\\", "-")


def plot_summary_by_region(df_metrics: pd.DataFrame, output_dir: str) -> None:
    """Summarize time asymmetry and PC1 by injection region."""
    os.makedirs(output_dir, exist_ok=True)
    for (spike_type, extract_from), df_group in df_metrics.groupby(
        ["spike_type", "extract_from"]
    ):
        df_group = df_group.dropna(subset=["injection region"])
        if df_group.empty:
            continue
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        generate_violin_plot(
            df_group,
            y_col="time_asymmetry",
            color_col="injection region",
            color_palette_dict=REGION_COLOR_MAPPER,
            ax=axes[0],
        )
        axes[0].set_title("Time asymmetry")
        generate_violin_plot(
            df_group,
            y_col="pc1_all",
            color_col="injection region",
            color_palette_dict=REGION_COLOR_MAPPER,
            ax=axes[1],
        )
        axes[1].set_title("PC1 projection (all)")
        fig.suptitle(f"{spike_type} | {extract_from}")
        filename = f"asymmetry_summary_{sanitize_filename(extract_from)}_{spike_type}"
        save_figure(
            fig,
            output_dir=output_dir,
            filename=filename,
            formats=("png",),
            bbox_inches="tight",
        )
        plt.close(fig)


def format_waveform_feature_name(col_name: str) -> str:
    """Format waveform asymmetry column names for display."""
    display = col_name.replace("waveform_time_asymmetry @ ", "")
    parts = display.split(" | ")
    extract_from = parts[0] if parts else display
    spike_type = parts[1] if len(parts) > 1 else ""

    spike_label = {
        "average": "average spike",
        "first": "first spike",
        "second": "second spike",
        "last": "last spike",
    }.get(spike_type, spike_type)
    header = f"{spike_label} dv/dt time asymmetry".strip()
    return f"{header} @ {extract_from}"


def plot_waveform_asymmetry_grid(df_metrics: pd.DataFrame, output_dir: str) -> None:
    """Generate a multi-panel scatter grid for waveform time asymmetry."""
    df_base = df_metrics[["ephys_roi_id", "injection region", "y"]].drop_duplicates(
        "ephys_roi_id"
    )
    df_features = df_metrics[
        ["ephys_roi_id", "extract_from", "spike_type", "time_asymmetry"]
    ].copy()
    df_features["feature_col"] = (
        "waveform_time_asymmetry @ "
        + df_features["extract_from"].astype(str)
        + " | "
        + df_features["spike_type"].astype(str)
    )
    df_wide = df_features.pivot_table(
        index="ephys_roi_id",
        columns="feature_col",
        values="time_asymmetry",
        aggfunc="first",
    )
    df_meta = df_base.merge(df_wide, on="ephys_roi_id", how="left")

    feature_cols = [col for col in df_wide.columns if col in df_meta.columns]
    features = [{col: format_waveform_feature_name(col)} for col in feature_cols]
    df_anova = anova_features(
        df_meta,
        features=feature_cols,
        cat_col="injection region",
        cont_col="y",
        adjust_p=True,
        anova_typ=2,
    )
    fig, _ = _generate_multi_feature_scatter_plots(
        df_meta=df_meta,
        features=features,
        df_anova=df_anova,
        filename="sup_fig_waveform_time_asymmetry",
        if_save_figure=False,
        n_cols=4,
    )
    save_figure(
        fig,
        output_dir=output_dir,
        filename="sup_fig_waveform_time_asymmetry",
        dpi=300,
        formats=("png", "svg"),
    )


def plot_pc_on_mesh(df_metrics: pd.DataFrame, output_dir: str) -> None:
    """Plot PC projections on the LC mesh for each spike type."""
    # Plot PC1 projections on the LC mesh for each spike type.
    os.makedirs(output_dir, exist_ok=True)
    mesh = load_mesh_from_s3()
    pc_variants = [
        ("pc1_waveform", "PC1 waveform"),
        ("pc1_dvdt", "PC1 dV/dt"),
        ("pc1_derived_ephys", "PC1 derived + ephys"),
        ("pc1_all", "PC1 all"),
    ]

    for spike_type, df_spike in df_metrics.groupby("spike_type"):
        extract_from_values = sorted(df_spike["extract_from"].dropna().unique())
        for pc_col, pc_label in pc_variants:
            df_variant = df_spike.dropna(subset=["x", "y", pc_col, "extract_from"])
            if df_variant.empty:
                continue
            n_panels = len(extract_from_values)
            n_cols = min(3, n_panels)
            n_rows = int(np.ceil(n_panels / n_cols))
            vmax = float(np.nanmax(np.abs(df_variant[pc_col])) or 1.0)

            fig, axes = plt.subplots(
                n_rows, n_cols, figsize=(4.5 * n_cols, 4.5 * n_rows), squeeze=False
            )
            sc = None
            for ax, extract_from in zip(axes.flat, extract_from_values):
                df_group = df_variant[df_variant["extract_from"] == extract_from]
                if df_group.empty:
                    ax.axis("off")
                    continue
                plot_mesh(ax, mesh, direction="sagittal", meshcol="lightgray")
                sc = ax.scatter(
                    df_group["x"],
                    df_group["y"],
                    c=df_group[pc_col],
                    cmap="RdBu_r",
                    vmin=-vmax,
                    vmax=vmax,
                    s=30,
                    edgecolor="black",
                    linewidth=0.3,
                    alpha=0.85,
                )
                ax.set_title(extract_from)
                ax.set_xlabel("Anterior-posterior (μm)")
                ax.set_ylabel("Dorsal-ventral (μm)")
                ax.set_aspect("equal")
                y_bottom, y_top = ax.get_ylim()
                ax.set_ylim(max(y_bottom, y_top), min(y_bottom, y_top))

            for ax in axes.flat[n_panels:]:
                ax.axis("off")

            fig.suptitle(
                f"{pc_label} on LC mesh | {spike_type}"
                "\nDerived = rise/fall/asymmetry + DEFAULT_EPHYS_FEATURES",
                fontsize=14,
            )
            if sc is not None:
                fig.colorbar(sc, ax=axes.ravel().tolist(), label=pc_label)
            filename = f"{pc_col}_mesh_grid_{spike_type}"
            save_figure(
                fig,
                output_dir=output_dir,
                filename=filename,
                formats=("png",),
                bbox_inches="tight",
            )
            plt.close(fig)


def run_anova_by_group(df_metrics: pd.DataFrame) -> pd.DataFrame:
    """Run ANCOVA for time asymmetry and PC1 across groups."""
    results = []
    for (spike_type, extract_from), df_group in df_metrics.groupby(
        ["spike_type", "extract_from"]
    ):
        features = ["time_asymmetry", "pc1_all"]
        df_anova = anova_features(
            df_group,
            features=features,
            cat_col="injection region",
            cont_col="y",
            adjust_p=True,
            anova_typ=2,
        )
        if df_anova.empty:
            continue
        df_anova["spike_type"] = spike_type
        df_anova["extract_from"] = extract_from
        results.append(df_anova)
    if not results:
        return pd.DataFrame()
    return pd.concat(results, ignore_index=True)


def main():
    """Entry point for waveform-derived asymmetry analysis."""
    logging.basicConfig(level=logging.INFO)
    set_plot_style(base_size=12, font_family="Helvetica")

    # 1) Load representative spikes per spike type and compute dv/dt timing metrics.
    example_count = 20
    spike_types = ["average", "first", "second", "last"]
    metrics_all = []

    for spike_type in spike_types:
        logger.info("Loading spikes: %s", spike_type)
        df_spikes = get_public_representative_spikes(spike_type=spike_type)
        df_spikes = drop_subthreshold(df_spikes)
        df_metrics, times_ms = compute_metrics_for_table(df_spikes, spike_type)
        metrics_all.append(df_metrics)

        # 2) Render a subset of diagnostic examples for manual inspection.
        example_pool = df_metrics.dropna(
            subset=["kick_idx", "peak_deriv_idx", "trough_deriv_idx"]
        )
        if example_pool.empty:
            example_rows = df_metrics.head(0)
        else:
            example_rows = example_pool.sample(
                n=min(example_count, len(example_pool)), random_state=42
            )
        output_dir = os.path.join(
            RESULTS_DIRECTORY, "figures", "asymmetry_waveform_examples"
        )
        for _, row in example_rows.iterrows():
            plot_example_spike(
                df_spikes,
                df_metrics,
                times_ms,
                str(row["ephys_roi_id"]),
                str(row["extract_from"]),
                spike_type,
                output_dir,
            )

    # 3) Merge all spike types into a single table and drop subthreshold stimuli.
    df_metrics_all = pd.concat(metrics_all, ignore_index=True)
    df_metrics_all = df_metrics_all[
        ~df_metrics_all["extract_from"].str.contains("subthreshold", case=False)
    ]
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    df_meta = df_meta.query(GLOBAL_FILTER).copy()
    default_ephys_cols = [list(item.keys())[0] for item in DEFAULT_EPHYS_FEATURES]
    ephys_cols = [col for col in default_ephys_cols if col in df_meta.columns]
    # 4) Join injection region metadata and default ephys features.
    df_metrics_all = df_metrics_all.merge(
        df_meta[["ephys_roi_id", "injection region", "y", "x", *ephys_cols]],
        on="ephys_roi_id",
        how="left",
    )
    df_metrics_all["extract_from"] = df_metrics_all["extract_from"].astype(str)

    # 5) Compute PCA variants and run ANCOVA per stimulus/spike type.
    df_metrics_all = df_metrics_all.dropna(subset=["injection region", "y"])
    df_metrics_all = add_pca_variants(df_metrics_all, ephys_cols)
    df_anova = run_anova_by_group(df_metrics_all)

    # 6) Generate summary plots, mesh projections, and a large asymmetry grid.
    plot_summary_by_region(
        df_metrics_all,
        output_dir=os.path.join(
            RESULTS_DIRECTORY, "figures", "asymmetry_waveform_summary"
        ),
    )
    plot_pc_on_mesh(
        df_metrics_all,
        output_dir=os.path.join(RESULTS_DIRECTORY, "figures", "asymmetry_pc_mesh"),
    )
    plot_waveform_asymmetry_grid(
        df_metrics_all,
        output_dir=os.path.join(
            RESULTS_DIRECTORY, "figures", "asymmetry_waveform_summary"
        ),
    )

    # 7) Save metrics and ANCOVA results.
    output_dir = os.path.join(RESULTS_DIRECTORY, "analysis")
    os.makedirs(output_dir, exist_ok=True)
    df_metrics_all.to_csv(
        os.path.join(output_dir, "waveform_asymmetry_metrics.csv"), index=False
    )
    df_anova.to_csv(
        os.path.join(output_dir, "waveform_asymmetry_anova.csv"), index=False
    )
    logger.info("Saved metrics to %s", output_dir)


if __name__ == "__main__":
    main()
