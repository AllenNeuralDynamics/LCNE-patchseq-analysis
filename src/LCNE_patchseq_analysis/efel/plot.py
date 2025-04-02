from matplotlib.gridspec import GridSpec
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def plot_sweep_raw(raw, df_features):
        time, trace, stimulus, stim_start, stim_end = raw["time"], raw["trace"], raw["stimulus"], raw["stim_start"], raw["stim_end"]
        
        time_interpolated = df_features["time"][0]
        
        # Plot the trace
        fig = plt.figure(figsize=(12, 6))
        gs = GridSpec(2, 1, height_ratios=[5, 1])
        ax = fig.add_subplot(gs[0, 0])
        ax_stimulus = fig.add_subplot(gs[1, 0], sharex=ax)

        ax.plot(time, trace, 'k-', lw=2)
        ax.axhline(df_features["voltage_base"][0], color="k", linestyle="--", label="voltage_base")
        
        if df_features["spike_count"][0][0] > 0:
                ax.plot(df_features["AP_begin_time"][0],
                        df_features["AP_begin_voltage"][0], "go",
                        label="AP_begin")
                ax.plot(df_features["peak_time"][0], df_features["peak_voltage"][0], "ro",
                        label="peak")
                ax.plot([time_interpolated[ind] for ind in df_features["min_AHP_indices"][0]],
                        df_features["min_AHP_values"][0], "ko",
                        label="min_AHP")

                # ["min_between_peaks_indices"]
                ax.plot([time_interpolated[ind] for ind in df_features["min_between_peaks_indices"][0]],
                        df_features["min_between_peaks_values"][0], "bo",
                        label="min_between_peaks")

                # Threshold
                threshold = efel.get_settings().Threshold
                ax.axhline(threshold, color="k", linestyle=":", label="threshold")

                # for i in range(df_features["spike_count"][0][0] - 1):
                #     ax.axhline(df_features["depolarized_base"][0][i], color="k", linestyle="--")

        # Plot sag, if "SubThresh" in stim_code
        if "SubThresh" in raw["stim_code"]:
            steady_state_voltage_stimend = df_features["steady_state_voltage_stimend"][0]
            ax.axhline(df_features["minimum_voltage"][0], color="gray", linestyle="--", label="minimum_voltage")
            ax.axhline(steady_state_voltage_stimend, color="deepskyblue", linestyle=":", label="steady_state_voltage_stimend")
            sag_amplitude = df_features["sag_amplitude"][0][0]
            ax.plot([stim_start, stim_start], 
                    [df_features["minimum_voltage"][0], df_features["minimum_voltage"][0] + sag_amplitude], 
                    color="deepskyblue", ls="-", label=f"sag_amplitude = {sag_amplitude:.2f}")
            voltage_deflection = df_features["voltage_deflection"][0][0]
            ax.plot([stim_start, stim_start], 
                    [steady_state_voltage_stimend, steady_state_voltage_stimend + (-voltage_deflection)], 
                     color="red", ls="-", label=f"voltage_deflection = {voltage_deflection:.2f}")

        # Add stimulus trace
        ax_stimulus.plot(time, stimulus, 'k-', lw=2)

        ax.set_xlabel('Time (ms)')
        ax.set_ylabel('V (mV)')
        meta = raw["meta"]
        ax.set_title(f'{meta.ephys_roi_id.values[0]} sweep #{meta.sweep_number.values[0]}, {meta.stimulus_code.values[0]}')
        ax.set_xlim(stim_start - max(3, (stim_end - stim_start) * 0.2), stim_end + max(100, (stim_end - stim_start) * 0.4))
        ax.legend(loc="best", fontsize=12)
        ax.label_outer()
        ax.grid(True)

        sns.despine()
        return fig



def plot_overlaid_spikes(raw, df_features):
    time, trace = raw["time"], raw["trace"]

    if df_features["spike_count"][0][0] == 0:
        return None
    
    # Plot individual spikes overlaid with some features

    sns.set_style("white")
    sns.set_context("talk")

    # peak_times = df_features["peak_time"][0]
    time_interpolated = df_features["time"][0]
    peak_times = [time_interpolated[ind] for ind in df_features["peak_indices"][0]]

    BEGIN_OFFSET = 2  # ms between begin_time to the eFEL AP_begin_time
    begin_times = [time_interpolated[ind] - BEGIN_OFFSET for ind in df_features["AP_begin_indices"][0]]
    end_times = begin_times[1:]  # From this begin to the start of tne next begin
    # end_times.append(stim_end)
    end_times.append(time_interpolated[df_features["min_between_peaks_indices"][0][-1]])

    n_spikes = len(peak_times)
    alphas = 0.8 * np.exp(-np.arange(n_spikes)/n_spikes)

    fig, axs = plt.subplots(1, 2, figsize=(13, 6))

    window = [-2, 6]  # Time window around a spike

    ax_v, ax_phase = axs

    # Plot the features for the first spike
    for i, (begin_time, end_time, peak_time) in reversed(list(enumerate(zip(begin_times, end_times, peak_times)))):
        # Get the trace around the spike
    #     start_time = peak_time + window[0]
    #     end_time = peak_time + window[1]
        idx = np.where((time >= begin_time) & (time <= end_time))[0]
        t = time[idx] - peak_time  # Center the time on the spike
        v = trace[idx]
        
        dvdt = np.gradient(v, t)

        if i > 0:
            # Only overlay the traces for following spikes with a lower opacity
            ax_v.plot(t, v, 'k', lw=0.5, alpha=alphas[i])
            
            # Plot the phase plot
            ax_phase.plot(v, dvdt, 'k', lw=0.5, alpha=alphas[i])
            continue

        # Plot the trace with some key features
        ax_v.plot(t, v, 'k', lw=3)
        ax_phase.plot(v, dvdt, 'k', lw=3)

        # Plot the peak
        ax_v.plot(0, df_features["peak_voltage"][0][i], "ro",
                label="peak", ms=10)

        # min_AHP
        ax_v.plot([time_interpolated[df_features["min_AHP_indices"][0][i]]] - peak_time,
                df_features["min_AHP_values"][0][i], "ko",
                label="min_AHP", ms=10)

        # AP_end
    #     ax_v.axvline(time_interpolated[df_features["AP_end_indices"][0][i]] - peak_time,
    #            color="b", label="AP_end")

        # AP_begin
        # Spike start is defined as where the first derivative of the voltage trace is higher than 10 V/s, for at least 5 points    
        t_begin = df_features["AP_begin_time"][0][i] - peak_time
        v_begin = df_features["AP_begin_voltage"][0][i]
        ax_v.plot(t_begin, v_begin, "go", label="AP_begin", ms=10)
        
        # AP_begin_width
        # https://efel.readthedocs.io/en/latest/eFeatures.html#ap-begin-width-ap1-begin-width-ap2-begin-width
        AP_begin_width = df_features["AP_begin_width"][0][i]        
        ax_v.plot([t_begin, t_begin + AP_begin_width],
                [v_begin, v_begin], "g-", label=f"AP_begin_width = {AP_begin_width:.2f}")

        # AP_width
        # https://efel.readthedocs.io/en/latest/eFeatures.html#ap-width
        threshold = efel.get_settings().Threshold
        threshold_time_idx = np.where(v >= threshold)[0][0]
        threshold_time = t[threshold_time_idx]
        AP_width = df_features["AP_width"][0][i]
        ax_v.plot(threshold_time, threshold, "ko", fillstyle="none", ms=10, label=f"Threshold $\equiv$ {threshold} mV")
        ax_v.plot([threshold_time, threshold_time + AP_width],
                [threshold, threshold], "k-", label=f"AP_width = {AP_width:.2f}")

        # AP_width_between_threshold
        # https://efel.readthedocs.io/en/latest/eFeatures.html#ap-width-between-threshold
        # Should be the same as AP_width for most cases

        # AP_duration_half_width
        # https://efel.readthedocs.io/en/latest/eFeatures.html#ap-duration-half-width
        half_rise_time = time_interpolated[df_features["AP_rise_indices"][0][i]] - peak_time
    #     half_fall_time = time_interpolated[df_features["AP_fall_indices"][0][i]] - peak_time
        half_voltage = (df_features["AP_begin_voltage"][0][i] + df_features["peak_voltage"][0][i]) / 2
        AP_duration_half_width = df_features["AP_duration_half_width"][0][i]
        ax_v.plot(half_rise_time, half_voltage, "mo", ms=10)
        ax_v.plot([half_rise_time, half_rise_time + AP_duration_half_width],
                [half_voltage, half_voltage], "m-", label=f"AP_duration_half_width = {AP_duration_half_width:.2f}")
        
        # Phase plot: phaseslope
        begin_ind = np.where(t >= t_begin)[0][0]
        ax_phase.plot(v[begin_ind], dvdt[begin_ind], 'go', ms=10, label="AP_begin")
        ax_phase.axhline(efel.get_settings().DerivativeThreshold, color="g", linestyle=":", label="Derivative threshold")
        
        # Phase plot: AP_phaseslope
        xx = np.linspace(v[begin_ind], v[begin_ind] + 10, 100)
        yy = dvdt[begin_ind] + (xx - v[begin_ind]) * df_features["AP_phaseslope"][0][i]
        ax_phase.plot(xx, yy, "g--", label=f"AP_phaseslope")
        
        # Phase plot: AP_peak_upstroke
        ax_phase.axhline(df_features["AP_peak_upstroke"][0][i], color="c", linestyle="--", label="AP_peak_upstroke")
        
        # Phase plot: AP_peak_downstroke
        ax_phase.axhline(df_features["AP_peak_downstroke"][0][i], color="darkblue", linestyle="--", label="AP_peak_downstroke")
        
    # For 1D array or single subplot
    ax_v.set_xlim(-BEGIN_OFFSET, 6)
    ax_v.set_xlabel('Time (ms)')
    ax_v.set_ylabel('V (mV)')
    ax_v.set_title(f'Overlaid spikes (n = {n_spikes})')
    ax_v.legend(loc="best", fontsize=12, title="1st spike features", title_fontsize=13)  # Outside the plot
    ax_v.grid(True)

    ax_phase.set_xlabel('Voltage (mV)')
    ax_phase.set_ylabel('dv/dt (mV/ms)')
    ax_phase.set_title('Phase Plots')
    ax_phase.legend(loc="best", fontsize=12, title="1st spike features", title_fontsize=13)  # Outside the plot
    ax_phase.grid(True)

    fig.tight_layout()
    sns.despine()
    
    return fig


def plot_sweep_summary(raw_traces, features_dict):
    
    ephys_roi_id = features_dict["df_sweeps"]["ephys_roi_id"][0]
    
    for raw_trace in raw_traces:
        sweep_number = raw_trace["sweep_number"][0]
        df_sweep_feature = features_dict["df_features_per_sweep"].query("sweep_number == @sweep_number")
        df_spike_feature = features_dict["df_features_per_spike"].query("sweep_number == @sweep_number")
        
        fig = plot_sweep_raw(raw_trace, df_sweep_feature)
        fig.savefig(f"{save_dir}/{ephys_roi_id}/{ephys_roi_id}_sweep_{sweep_number}.png")
