"""Lightweight panel app for navigating NWB files.
Run this in command line:
    panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org
"""

import panel as pn
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import numpy as np
import glob

sampling_rate = 50000
dt_ms = 1 / sampling_rate * 1000

# ---- Data Loading Module ----
def load_nwb_data(file_path):
    """
    Loads an NWB file and returns the nwbfile object.
    """
    nwbfile = NWBHDF5IO(file_path, 'r').read()
    return nwbfile

# ---- Plotting Function ----
def update_plot(n, nwb):
    """
    Extracts a slice of data from the NWB file and returns a matplotlib figure.
    Adjust the data extraction logic based on your NWB file structure.
    """

    # Using nwb
    trace = nwb.acquisition[f"data_{n:05}_AD0"].data[:]
    stimulus = nwb.stimulus[f"data_{n:05}_DA0"].data[:]
    time = np.arange(len(trace)) * dt_ms
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), dpi=100)
    ax[0].plot(time, trace)
    ax[0].set_title(f'Sweep number {n}')
    
    ax[1].plot(time, stimulus)
    plt.close(fig)  # Prevents duplicate display
    return fig
  

# ---- Main Panel App Layout ----
def main():
    
    RAW_DIRECTORY = "/root/capsule/data/LCNE-patchseq-ephys-raw"
    ephys_roi_id = "1410790193"
    raw_path_this = f"{RAW_DIRECTORY}/Ephys_Roi_Result_{ephys_roi_id}"
    
    nwbs = glob.glob(f"{raw_path_this}/*.nwb")
    
    # Load NWB data (adjust the file path as needed)
    nwb_data = load_nwb_data(nwbs[1])
    
    # Define a slider widget. Adjust the range based on your NWB data dimensions.
    n_sweeps = len(nwb_data.acquisition)
    slider = pn.widgets.IntSlider(name='Data Slice', start=0, end=n_sweeps-1, value=0)
    
    # Bind the slider value to the update_plot function.
    # pn.bind creates a reactive function that updates the plot whenever the slider changes.
    plot_panel = pn.bind(update_plot, n=slider, nwb=nwb_data)
    
    # Compose the layout: place the slider above the plot.
    layout = pn.Column(slider, pn.panel(plot_panel, sizing_mode='stretch_both'))
    
    # Make the panel servable if running with 'panel serve'
    return layout

layout = main()
layout.servable()
