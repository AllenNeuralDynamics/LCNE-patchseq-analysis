import logging
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER, RESULTS_DIRECTORY
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures import GLOBAL_FILTER, set_plot_style
from LCNE_patchseq_analysis.figures.util import generate_violin_plot, save_figure
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes
from LCNE_patchseq_analysis.population_analysis.anova import anova_features

logger = logging.getLogger(__name__)


def find_local_maxima(values: np.ndarray) -> np.ndarray:
    if values.size < 3:
        return np.array([], dtype=int)
    return np.where((values[1:-1] > values[:-2]) & (values[1:-1] > values[2:]))[0] + 1


def compute_spike_metrics(times_ms: np.ndarray, voltages: np.ndarray) -> dict:
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
    return df_metrics, np.vstack(waveforms), np.vstack(derivatives), times_ms


def add_pca_projection(
    df_metrics: pd.DataFrame,
    waveforms: np.ndarray,
    derivatives: np.ndarray,
) -> pd.DataFrame:
    metric_cols = ["rise_time_ms", "fall_time_ms", "time_asymmetry"]
    metric_values = df_metrics[metric_cols].to_numpy(dtype=float)
    features = np.column_stack([waveforms, derivatives, metric_values])
    valid_mask = np.isfinite(features).all(axis=1)

    pc1 = np.full(len(df_metrics), np.nan)
    if valid_mask.sum() > 2:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features[valid_mask])
        pca = PCA(n_components=1)
        pc1_vals = pca.fit_transform(features_scaled).ravel()
        pc1[valid_mask] = pc1_vals
    df_metrics["pc1"] = pc1
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
    key = (str(ephys_roi_id), extract_from)
    if key not in df_spikes.index:
        return
    voltage = df_spikes.loc[key].to_numpy(dtype=float)
    metrics = df_metrics.query(
        "ephys_roi_id == @ephys_roi_id and extract_from == @extract_from and spike_type == @spike_type"
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
            label="kick",
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


def sanitize_filename(text: str) -> str:
    return text.replace(",", "").replace(" ", "_").replace("/", "-").replace("\\", "-")


def plot_summary_by_region(df_metrics: pd.DataFrame, output_dir: str) -> None:
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
            y_col="pc1",
            color_col="injection region",
            color_palette_dict=REGION_COLOR_MAPPER,
            ax=axes[1],
        )
        axes[1].set_title("PC1 projection")
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


def run_anova_by_group(df_metrics: pd.DataFrame) -> pd.DataFrame:
    results = []
    for (spike_type, extract_from), df_group in df_metrics.groupby(
        ["spike_type", "extract_from"]
    ):
        features = ["time_asymmetry", "pc1"]
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
    logging.basicConfig(level=logging.INFO)
    set_plot_style(base_size=12, font_family="Helvetica")

    example_count = 20
    spike_types = ["average", "first", "second", "last"]
    metrics_all = []

    for spike_type in spike_types:
        logger.info("Loading spikes: %s", spike_type)
        df_spikes = get_public_representative_spikes(spike_type=spike_type)
        df_metrics, waveforms, derivatives, times_ms = compute_metrics_for_table(
            df_spikes, spike_type
        )
        df_metrics = add_pca_projection(df_metrics, waveforms, derivatives)
        metrics_all.append(df_metrics)

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
                row["ephys_roi_id"],
                row["extract_from"],
                spike_type,
                output_dir,
            )

    df_metrics_all = pd.concat(metrics_all, ignore_index=True)
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    df_meta = df_meta.query(GLOBAL_FILTER).copy()
    df_metrics_all = df_metrics_all.merge(
        df_meta[["ephys_roi_id", "injection region", "y"]],
        on="ephys_roi_id",
        how="left",
    )
    df_metrics_all["extract_from"] = df_metrics_all["extract_from"].astype(str)

    df_metrics_all = df_metrics_all.dropna(subset=["injection region", "y"])
    df_anova = run_anova_by_group(df_metrics_all)
    plot_summary_by_region(
        df_metrics_all,
        output_dir=os.path.join(
            RESULTS_DIRECTORY, "figures", "asymmetry_waveform_summary"
        ),
    )
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
