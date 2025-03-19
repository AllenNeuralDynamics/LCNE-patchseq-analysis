"""Lightweight panel app for navigating NWB files.
Run this in command line:
    panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org
"""

import matplotlib.pyplot as plt
import numpy as np
import panel as pn

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

    fig, ax = plt.subplots(2, 1, figsize=(6, 4), gridspec_kw={"height_ratios": [3, 1]})
    ax[0].plot(time, trace)
    ax[0].set_title(f"Sweep number {sweep}")
    ax[0].set(ylabel="Vm (mV)")

    ax[1].plot(time, stimulus)
    ax[1].set(xlabel="Time (ms)", ylabel="I (pA)")
    ax[0].label_outer()

    plt.close(fig)  # Prevents duplicate display
    return fig



# ---- Main Panel App Layout ----
def main():
    """main app"""

    pn.config.throttled = False

    # Load the NWB file.
    raw = PatchSeqNWB(ephys_roi_id="1410790193")

    # Define a slider widget. Adjust the range based on your NWB data dimensions.
    slider = pn.widgets.IntSlider(name="Sweep number", start=0, end=raw.n_sweeps - 1, value=0)

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
        stylesheets=[":host .tabulator {font-size: 12px;}"],
    )

    # --- Two-Way Synchronization between Slider and Table ---
    # When the user selects a row in the table, update the slider.
    def update_slider_from_table(event):
        """table --> slider"""
        if event.new:
            # event.new is a list of selected row indices; assume single selection.
            selected_index = event.new[0]
            new_sweep = raw.df_sweeps.loc[selected_index, "sweep_number"]
            slider.value = new_sweep

    tab.param.watch(update_slider_from_table, "selection")

    # When the slider value changes, update the table selection.
    def update_table_selection(event):
        """Update slider --> table"""
        new_val = event.new
        row_index = raw.df_sweeps.index[raw.df_sweeps["sweep_number"] == new_val].tolist()
        tab.selection = row_index

    slider.param.watch(update_table_selection, "value")
    # --- End Synchronization ---

    # --- Error Message if Sweep Not Found ---
    def get_error_message(sweep):
        """Get error message"""
        if sweep not in raw.df_sweeps["sweep_number"].values:
            return "<span style='color:red;'>Sweep number not found in the jsons!</span>"
        return ""

    error_msg = pn.bind(get_error_message, slider.param.value)
    error_msg_panel = pn.pane.Markdown(error_msg, width=600, height=30)
    # --- End Error Message ---

    # Compose the layout: error message, slider, and plot on the left; table on the right.
    left_col = pn.Column(slider, error_msg_panel, mpl_pane)
    layout = pn.Row(
        pn.Column(
            pn.pane.Markdown(
                "# Patch-seq Ephys Data Navigator\n"
                "Use the slider to navigate through the sweeps in the NWB file."
            ),
            left_col,
        ),
        pn.Column(
            pn.pane.Markdown("## Metadata from jsons"),
            tab,
        ),
    )

    # Make the panel servable if running with 'panel serve'
    return layout


layout = main()
layout.servable()
