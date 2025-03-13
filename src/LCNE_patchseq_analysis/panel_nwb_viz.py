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

    pn.config.throttled = True
    
    # Load the NWB file.
    raw = PatchSeqNWB(ephys_roi_id="1410790193")

    # Define a slider widget. Adjust the range based on your NWB data dimensions.
    slider = pn.widgets.IntSlider(
        name="Sweep number", start=0, end=raw.n_sweeps - 1, value=0
    )

    text_panel = pn.pane.Markdown("# Patch-seq Ephys Data Navigator\nUse the slider to navigate through the sweeps in the NWB file.")

    # Bind the slider value to the update_plot function.
    plot_panel = pn.bind(update_plot, raw=raw, sweep=slider.param.value)
    mpl_pane = pn.pane.Matplotlib(plot_panel, dpi=400, width=600, height=400)

    # Create a Tabulator widget for the DataFrame with row selection enabled.
    tab = pn.widgets.Tabulator(
        raw.df_sweeps[
            [
                "sweep_number",
                "stimulus_code_ext",
                "stimulus_name",
                "stimulus_amplitude",
                "passed",
                "num_spikes",
                "stimulus_start_time",
                "stimulus_duration",
                "tags",
                "reasons",
                "stimulus_code",
            ]
        ],
        hidden_columns=["stimulus_code"],
        selectable=1,
        disabled=True,  # Not editable
        frozen_columns=["sweep_number"],
        header_filters=True,
        show_index=False,
        height=700,
        width=1000,
        groupby=["stimulus_code"],
        stylesheets=[":host .tabulator {font-size: 12px;}"]
    )

    # --- Two-Way Synchronization between Slider and Table ---
    # When the user selects a row in the table, update the slider.
    def update_slider_from_table(event):
        if event.new:
            # event.new is a list of selected row indices; assume single selection.
            selected_index = event.new[0]
            new_sweep = raw.df_sweeps.loc[selected_index, 'sweep_number']
            slider.value = new_sweep

    tab.param.watch(update_slider_from_table, 'selection')

    # When the slider value changes, update the table selection.
    def update_table_selection(event):
        new_val = event.new
        row_index = raw.df_sweeps.index[raw.df_sweeps['sweep_number'] == new_val].tolist()
        tab.selection = row_index

    slider.param.watch(update_table_selection, 'value')
    # --- End Synchronization ---

    # Compose the layout: place the slider and plot on the left, table on the right.
    layout = pn.Column(
        text_panel,
        pn.Row(pn.Column(slider, mpl_pane), tab),
    )

    # Make the panel servable if running with 'panel serve'
    return layout

layout = main()
layout.servable()
