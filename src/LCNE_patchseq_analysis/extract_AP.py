import os
import gc
import h5py
import json
import numpy as np
import matplotlib.pyplot as plt

# Directories (Modify paths as needed)
CODE_DIRECTORY = "/root/capsule/code/"
DATA_DIRECTORY = "/root/capsule/data/JeremiahYCohen/LC-NE/"
RESULTS_DIRECTORY = "/root/capsule/results/"


def find_spikes(x, threshold=10, half_window_size=15):
    """
    Find indices in x that are local maxima within a window and exceed the threshold.
    """
    x = np.asarray(x)
    window_size = half_window_size * 2 + 1

    # If x is too short, return an empty array.
    if len(x) < window_size:
        return np.array([], dtype=int)

    # Create sliding windows of length window_size.
    # Each row is a window of consecutive values from x.
    windows = np.lib.stride_tricks.sliding_window_view(x, window_size)

    # For each window, check if the maximum value occurs in the center.
    # np.argmax returns the first occurrence of the maximum, matching R's behavior.
    # In Python, indexing is 0-based so the center is at index half_window_size.
    local_maxima = np.argmax(windows, axis=1) == half_window_size

    # The candidate indices in x correspond to the center of each window.
    # They range from half_window_size to len(x) - half_window_size.
    candidate_indices = np.arange(half_window_size, len(x) - half_window_size)

    # Select indices where the center is a local maximum and the value exceeds the threshold.
    spikes = candidate_indices[local_maxima & (x[candidate_indices] > threshold)]
    return spikes


# Load data files
def get_data_files():
    files = [f for f in os.listdir(DATA_DIRECTORY) if "Roi" in f]
    nwb_files = []
    json_files = []

    for folder in files:
        folder_path = os.path.join(DATA_DIRECTORY, folder)

        nwb_files.extend(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".nwb") and "spikes" not in f
            ]
        )
        json_files.extend(
            [
                os.path.join(folder_path, f)
                for f in os.listdir(folder_path)
                if f.endswith(".json") and "EPHYS_QC_V3" in f and "input" in f
            ]
        )

    return nwb_files, json_files


# Extract APs from nwb files
def extract_aps(nwb_files, json_files):
    all_aps = {}

    for i, (nwb_file, json_file) in enumerate(zip(nwb_files, json_files)):
        print(f"Processing cell {i+1}/{len(nwb_files)}...")

        # Get metadata
        with open(json_file, "r") as jfile:
            session_data = json.load(jfile)
            trial_numbers = [s["sweep_number"] for s in session_data["sweep_features"]]

        # Get spike waveforms
        with h5py.File(nwb_file, "r") as hdf:
            acquisition = hdf["/acquisition/"]

            spikes_per_trial = []
            raw_per_sweep = []
            for trial in trial_numbers:
                sweep_name = list(acquisition)[trial]
                this_acquisition = acquisition[sweep_name]
                # Sanity check
                if str(trial) not in list(acquisition)[trial]:
                    print(f"Trial {trial} not found in acquisition data.")
                    continue

                raw_voltage = np.array(this_acquisition["data"])
                spike_times = find_spikes(raw_voltage)

                # Save raw traces
                raw_per_sweep.append(raw_voltage)

                # Save spike waveforms
                spike_waveforms = [
                    raw_voltage[t - 500:t + 1500]
                    for t in spike_times
                    if t > 500 and t + 1500 < len(raw_voltage)
                ]
                if spike_waveforms:
                    spikes_per_trial.append(np.array(spike_waveforms))

            all_aps[nwb_file] = {
                "spikes_per_trial": spikes_per_trial,
                "raw_per_sweep": raw_per_sweep,
            }

    return all_aps


# Compute mean APs
def compute_mean_aps(all_aps):
    mean_aps = {}
    for cell, all_ap in all_aps.items():
        spikes = all_ap["spikes_per_trial"]
        if spikes:
            all_spikes = np.vstack(spikes)
            mean_aps[cell] = np.mean(all_spikes, axis=0)
        else:
            mean_aps[cell] = np.full((2001,), np.nan)
    return mean_aps


# Plot mean APs
def plot_mean_aps(mean_aps):
    fig, ax = plt.subplots(figsize=(10, 6))
    for mean_ap in mean_aps.values():
        ax.plot(mean_ap, color="black", alpha=0.5)
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (mV)")
    ax.set_title("Mean Action Potentials")

    fig.savefig(RESULTS_DIRECTORY + "mean_action_potentials.png", dpi=700)


# Main script execution
def main():
    gc.collect()
    nwb_files, json_files = get_data_files()
    all_aps = extract_aps(nwb_files[:3], json_files[:3])
    mean_aps = compute_mean_aps(all_aps)
    plot_mean_aps(mean_aps)

    # Save results
    np.save(os.path.join(RESULTS_DIRECTORY, "all_aps.npy"), all_aps)
    np.save(os.path.join(RESULTS_DIRECTORY, "mean_aps.npy"), mean_aps)
    print("Analysis complete. Mean APs saved.")


if __name__ == "__main__":
    main()
