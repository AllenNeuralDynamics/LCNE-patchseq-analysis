"""Lightweight panel app for navigating NWB files.
Run this in command line:
    panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org
"""

import panel as pn
import matplotlib.pyplot as plt
import numpy as np

from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB

# ---- Plotting Function ----
def update_plot(raw, sweep):
    """
    Extracts a slice of data from the NWB file and returns a matplotlib figure.
    Adjust the data extraction logic based on your NWB file structure.
    """

    # Using nwb
    trace = raw.get_raw_trace(sweep)
    stimulus = raw.get_stimulus(sweep)
    time = np.arange(len(trace)) * raw.dt_ms
    
    print(time)
    fig, ax = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={'height_ratios': [3, 1]})
    ax[0].plot(time, trace)
    ax[0].set_title(f'Sweep number {sweep}')
    ax[0].set(ylabel='Vm (mV)')
    
    ax[1].plot(time, stimulus)
    ax[1].set(xlabel='Time (ms)', ylabel='I (pA)')
    ax[0].label_outer()
    
    plt.close(fig)  # Prevents duplicate display
    return fig

# Function to style the DataFrame, highlighting the row with the selected sweep_number.
def show_df_with_highlight(df, selected_sweep):
    def highlight_row(row):
        return ['background-color: yellow' if row.sweep_number == selected_sweep else '' 
                for _ in row.index]
    return df.style.apply(highlight_row, axis=1)

# ---- Main Panel App Layout ----
def main():

    # Define a slider widget. Adjust the range based on your NWB data dimensions.
    raw = PatchSeqNWB(ephys_roi_id="1410790193")  # Load the NWB file

    slider = pn.widgets.IntSlider(name='Sweep number', start=0, end=raw.n_sweeps-1, value=0)

    text_panel = pn.pane.Markdown("# Patch-seq Ephys Data\nUse the slider to navigate through the sweeps in the NWB file.",)

    # Bind the slider value to the update_plot function.
    plot_panel = pn.bind(update_plot, raw=raw, sweep=slider.param.value_throttled)
    mpl_pane = pn.pane.Matplotlib(plot_panel, dpi=400, width=600, height=400)

    panel_df = pn.bind(show_df_with_highlight, raw.df_sweeps, selected_sweep=slider.param.value_throttled)

    # Compose the layout: place the slider above the plot.
    layout = pn.Column(
        text_panel,
        pn.Row(pn.Column(slider, mpl_pane), pn.panel(panel_df, sizing_mode="stretch_width")),
    )

    # Make the panel servable if running with 'panel serve'
    return layout

layout = main()
layout.servable()
