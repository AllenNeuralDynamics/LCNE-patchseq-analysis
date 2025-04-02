import os
from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

from LCNE_patchseq_analysis import TIME_STEP

sns.set_style("white")
sns.set_context("talk")


def plot_sweep_raw(raw_trace, df_sweep_meta, df_sweep_feature, df_spike_feature):

    (time, trace, stimulus, stim_start, stim_end) = (
        raw_trace["T"], raw_trace["V"], raw_trace["stimulus"], raw_trace["stim_start"][0], raw_trace["stim_end"][0])

    time_interpolated = time  # This only works for non-interpolated case. (TIME_STEP = interp_step)

    # Plot the trace
    fig = plt.figure(figsize=(12, 6))
    gs = GridSpec(2, 1, height_ratios=[5, 1])
    ax = fig.add_subplot(gs[0, 0])
    ax_stimulus = fig.add_subplot(gs[1, 0], sharex=ax)

    ax.plot(time, trace, 'k-', lw=2)
    ax.axhline(df_sweep_feature["voltage_base"], color="k", linestyle="--", label="voltage_base")

    if df_sweep_feature["spike_count"] > 0:
        ax.plot(df_spike_feature["AP_begin_time"],
                df_spike_feature["AP_begin_voltage"], "go",
                label="AP_begin")
        ax.plot(df_spike_feature["peak_time"], df_spike_feature["peak_voltage"], "ro",
                label="peak")
        ax.plot([time_interpolated[ind] for ind in df_spike_feature["min_AHP_indices"].astype(int)],
                df_spike_feature["min_AHP_values"], "ko",
                label="min_AHP")

        # ["min_between_peaks_indices"]
        ax.plot([time_interpolated[ind] for ind in df_spike_feature["min_between_peaks_indices"].astype(int)],
                df_spike_feature["min_between_peaks_values"], "bo",
                label="min_between_peaks")

        # for i in range(df_features["spike_count"][0][0] - 1):
        #     ax.axhline(df_features["depolarized_base"][0][i], color="k", linestyle="--")

    # Plot sag, if "SubThresh" in stim_code
    if "SubThresh" in df_sweep_meta["stimulus_code"].values[0]:
        steady_state_voltage_stimend = df_sweep_feature["steady_state_voltage_stimend"]

        ax.axhline(df_sweep_feature["minimum_voltage"], color="gray",
                   linestyle="--", label="minimum_voltage")
        ax.axhline(steady_state_voltage_stimend, color="deepskyblue",
                   linestyle=":", label="steady_state_voltage_stimend")

        sag_amplitude = df_sweep_feature["sag_amplitude"]
        ax.plot([stim_start, stim_start],
                [df_sweep_feature["minimum_voltage"],
                 df_sweep_feature["minimum_voltage"] + sag_amplitude],
                color="deepskyblue", ls="-", label=f"sag_amplitude = {sag_amplitude:.2f}")
        voltage_deflection = df_sweep_feature["voltage_deflection"]
        ax.plot([stim_start, stim_start],
                [steady_state_voltage_stimend,
                    steady_state_voltage_stimend + (-voltage_deflection)],
                color="red", ls="-", label=f"voltage_deflection = {voltage_deflection:.2f}")

    # Add stimulus trace
    ax_stimulus.plot(time, stimulus, 'k-', lw=2)

    ax.set_xlabel('Time (ms)')
    ax.set_ylabel('V (mV)')
    ax.set_title(
        f'{df_sweep_meta.ephys_roi_id.values[0]} sweep #{df_sweep_meta.sweep_number.values[0]}, {df_sweep_meta.stimulus_code.values[0]}')
    ax.set_xlim(stim_start - max(3, (stim_end - stim_start) * 0.2),
                stim_end + max(100, (stim_end - stim_start) * 0.4))
    ax.legend(loc="best", fontsize=12)
    ax.label_outer()
    ax.grid(True)

    sns.despine()
    return fig


def plot_overlaid_spikes(spike_this, df_spike_feature, efel_settings, width_scale: float = 3, beta: float = 2):

    n_spikes = len(spike_this)
    t = spike_this.columns
    peak_time_idx_in_t = np.argmin(np.abs(t - 0))

    alphas = 1.0 * np.exp(-beta * np.arange(n_spikes)/n_spikes)
    widths = width_scale * np.exp(-beta * np.arange(n_spikes)/n_spikes)

    fig, axs = plt.subplots(1, 2, figsize=(13, 6))
    ax_v, ax_phase = axs

    # Plot the features for the first spike
    for i in range(n_spikes):
        v = spike_this.query("spike_idx == @i").values[0]
        peak_time = df_spike_feature["peak_time"].loc[i]
        # This only works for non-interpolated case. (TIME_STEP = interp_step)
        peak_time_idx_in_raw = int(peak_time / TIME_STEP)
        dvdt = np.gradient(v, t)

        # -- For all spikes except the first one --
        if i > 0:
            # Only overlay the traces for following spikes with a lower opacity
            ax_v.plot(t, v, 'k', lw=widths[i], alpha=alphas[i])

            # Plot the phase plot
            ax_phase.plot(v, dvdt, 'k', lw=widths[i], alpha=alphas[i])
            continue

        # -- Highlight the first spike --
        # Plot the trace with some key features
        ax_v.plot(t, v, 'k', lw=widths[0])
        ax_phase.plot(v, dvdt, 'k', lw=widths[0])

        # Plot the peak
        ax_v.plot(0, df_spike_feature["peak_voltage"].loc[i], "ro",
                  label="peak", ms=10)

        # min_AHP
        ax_v.plot(t[df_spike_feature["min_AHP_indices"].loc[0].astype(int) 
                    - peak_time_idx_in_raw 
                    + peak_time_idx_in_t],
                  df_spike_feature["min_AHP_values"].loc[0], "ko",
                  label="min_AHP", ms=10)

        # AP_end
    #     ax_v.axvline(time_interpolated[df_features["AP_end_indices"][0][i]] - peak_time,
    #            color="b", label="AP_end")

        # AP_begin
        # Spike start is defined as where the first derivative of the voltage trace is higher than 10 V/s, for at least 5 points
        t_begin = df_spike_feature["AP_begin_time"].loc[i] - peak_time
        v_begin = df_spike_feature["AP_begin_voltage"].loc[i]
        ax_v.plot(t_begin, v_begin, "go", label="AP_begin", ms=10)

        # AP_begin_width
        # https://efel.readthedocs.io/en/latest/eFeatures.html#ap-begin-width-ap1-begin-width-ap2-begin-width
        AP_begin_width = df_spike_feature["AP_begin_width"].loc[i]
        ax_v.plot([t_begin, t_begin + AP_begin_width],
                  [v_begin, v_begin], "g-", label=f"AP_begin_width = {AP_begin_width:.2f}")

        # AP_width
        # https://efel.readthedocs.io/en/latest/eFeatures.html#ap-width
        threshold = efel_settings["Threshold"]
        threshold_time_idx = np.where(v >= threshold)[0][0]
        threshold_time = t[threshold_time_idx]
        AP_width = df_spike_feature["AP_width"].loc[i]
        ax_v.plot(threshold_time, threshold, "ko", fillstyle="none",
                  ms=10, label=f"Threshold $\equiv$ {threshold} mV")
        ax_v.plot([threshold_time, threshold_time + AP_width],
                  [threshold, threshold], "k-", label=f"AP_width = {AP_width:.2f}")

        # AP_width_between_threshold
        # https://efel.readthedocs.io/en/latest/eFeatures.html#ap-width-between-threshold
        # Should be the same as AP_width for most cases

        # AP_duration_half_width
        # https://efel.readthedocs.io/en/latest/eFeatures.html#ap-duration-half-width
        half_rise_time = t[df_spike_feature["AP_rise_indices"].loc[0].astype(
            int) - peak_time_idx_in_raw + peak_time_idx_in_t]
    #     half_fall_time = time_interpolated[df_features["AP_fall_indices"][0][i]] - peak_time
        half_voltage = (df_spike_feature["AP_begin_voltage"].loc[i] +
                        df_spike_feature["peak_voltage"].loc[i]) / 2
        AP_duration_half_width = df_spike_feature["AP_duration_half_width"].loc[i]
        ax_v.plot(half_rise_time, half_voltage, "mo", ms=10)
        ax_v.plot([half_rise_time, half_rise_time + AP_duration_half_width],
                  [half_voltage, half_voltage], "m-", label=f"AP_duration_half_width = {AP_duration_half_width:.2f}")

        # Phase plot: phaseslope
        begin_ind = np.where(t >= t_begin)[0][0]
        ax_phase.plot(v[begin_ind], dvdt[begin_ind], 'go', ms=10, label="AP_begin")
        ax_phase.axhline(efel_settings["DerivativeThreshold"],
                         color="g", linestyle=":", label="Derivative threshold")

        # Phase plot: AP_phaseslope
        xx = np.linspace(v[begin_ind], v[begin_ind] + 10, 100)
        yy = dvdt[begin_ind] + (xx - v[begin_ind]) * df_spike_feature["AP_phaseslope"].loc[i]
        ax_phase.plot(xx, yy, "g--", label=f"AP_phaseslope")

        # Phase plot: AP_peak_upstroke
        ax_phase.axhline(df_spike_feature["AP_peak_upstroke"].loc[i],
                         color="c", linestyle="--", label="AP_peak_upstroke")

        # Phase plot: AP_peak_downstroke
        ax_phase.axhline(df_spike_feature["AP_peak_downstroke"].loc[i],
                         color="darkblue", linestyle="--", label="AP_peak_downstroke")

    # For 1D array or single subplot
    ax_v.set_xlim(-2, 6)
    ax_v.set_xlabel('Time (ms)')
    ax_v.set_ylabel('V (mV)')
    ax_v.set_title(f'Overlaid spikes (n = {n_spikes})')
    ax_v.legend(loc="best", fontsize=12, title="1st spike features",
                title_fontsize=13)  # Outside the plot
    ax_v.grid(True)

    ax_phase.set_xlabel('Voltage (mV)')
    ax_phase.set_ylabel('dv/dt (mV/ms)')
    ax_phase.set_title('Phase Plots')
    ax_phase.legend(loc="best", fontsize=12, title="1st spike features",
                    title_fontsize=13)  # Outside the plot
    ax_phase.grid(True)

    fig.tight_layout()
    sns.despine()

    return fig


def plot_sweep_summary(raw_traces, features_dict, save_dir):

    ephys_roi_id = features_dict["df_sweeps"]["ephys_roi_id"][0]

    os.makedirs(f"{save_dir}/{ephys_roi_id}", exist_ok=True)

    for raw_trace in raw_traces:
        sweep_number = raw_trace["sweep_number"][0]
        df_sweep_feature = features_dict["df_features_per_sweep"].loc[sweep_number]
        has_spikes = df_sweep_feature["spike_count"] > 0

        df_spike_feature = features_dict["df_features_per_spike"].loc[
            sweep_number] if has_spikes else None
        df_sweep_meta = features_dict["df_sweeps"].query("sweep_number == @sweep_number")

        fig_sweep = plot_sweep_raw(raw_trace, df_sweep_meta, df_sweep_feature, df_spike_feature)
        fig_sweep.savefig(
            f"{save_dir}/{ephys_roi_id}/{ephys_roi_id}_sweep_{sweep_number}.png", dpi=400)

        if has_spikes:
            spike_this = features_dict["df_spike_waveforms"].query("sweep_number == @sweep_number")
            fig_spikes = plot_overlaid_spikes(
                spike_this, df_spike_feature, features_dict["efel_settings"][0], width_scale=3, beta=2)
            fig_spikes.savefig(
                f"{save_dir}/{ephys_roi_id}/{ephys_roi_id}_spikes_{sweep_number}_spikes.png", dpi=400)
