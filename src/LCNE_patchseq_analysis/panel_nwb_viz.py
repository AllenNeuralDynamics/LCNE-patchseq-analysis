"""Lightweight panel app for navigating NWB files.
Run this in command line:
    panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org
"""

import panel as pn
import matplotlib.pyplot as plt
from pynwb import NWBHDF5IO
import numpy as np
import glob
import h5py

sampling_rate = 50000
dt_ms = 1 / sampling_rate * 1000

# ---- Data Loading Module ----
def load_nwb_data(file_path):
    """
    Loads an NWB file and returns the nwbfile object.
    """
    nwbfile = NWBHDF5IO(file_path, 'r').read()
    return nwbfile

def load_h5py_data(file_path):
    """
    Loads an h5py file and returns the h5py object.
    """
    h5pyfile = h5py.File(file_path, 'r')
    return h5pyfile

# ---- Plotting Function ----
def update_plot(n, hdf):
    """
    Extracts a slice of data from the NWB file and returns a matplotlib figure.
    Adjust the data extraction logic based on your NWB file structure.
    """

    # Using nwb
    trace = np.array(hdf[f"/acquisition/data_{n:05}_AD0/data"])
    stimulus = np.array(hdf[f"/stimulus/presentation/data_{n:05}_DA0/data"])
    time = np.arange(len(trace)) * dt_ms
    
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(time, trace)
    ax[0].set_title(f'Sweep number {n}')
    ax[0].set(ylabel='Vm (mV)')
    
    ax[1].plot(time, stimulus)
    ax[1].set(xlabel='Time (ms)', ylabel='I (pA)')
    ax[0].label_outer()
    
    plt.close(fig)  # Prevents duplicate display
    return fig
  

# ---- Main Panel App Layout ----
def main():
    
    RAW_DIRECTORY = "/root/capsule/data/LCNE-patchseq-ephys-raw"
    ephys_roi_id = "1410790193"
    raw_path_this = f"{RAW_DIRECTORY}/Ephys_Roi_Result_{ephys_roi_id}"
    
    nwbs = glob.glob(f"{raw_path_this}/*spikes.nwb")
    
    # Load NWB data (adjust the file path as needed)
    hdf = load_h5py_data(nwbs[0])
    
    # Define a slider widget. Adjust the range based on your NWB data dimensions.
    n_sweeps = len(hdf["acquisition"])
    slider = pn.widgets.IntSlider(name='Sweep number', start=0, end=n_sweeps-1, value=0)
    
    # Bind the slider value to the update_plot function.
    # pn.bind creates a reactive function that updates the plot whenever the slider changes.
    plot_panel = pn.bind(update_plot, n=slider, hdf=hdf)
    
    mpl_pane = pn.pane.Matplotlib(plot_panel, dpi=400, width=600, height=400)
    
    # Compose the layout: place the slider above the plot.
    layout = pn.Column(slider, mpl_pane)
    
    # Make the panel servable if running with 'panel serve'
    return layout

layout = main()
layout.servable()
