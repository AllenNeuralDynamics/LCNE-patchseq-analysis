"""Lightweight panel app for navigating NWB files.
Run this in command line:
    panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org
"""

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import param

from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
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


def highlight_selected_rows(row, highlight_subset, color, fields=None):
    """Highlight rows based on a subset of values.

    If fields is None, highlight the entire row.
    """
    style = [""] * len(row)
    if row["sweep_number"] in highlight_subset:
        if fields is None:
            return [f"background-color: {color}"] * len(row)
        else:
            for field in fields:
                style[list(row.keys()).index(field)] = f"background-color: {color}"
    return style


# --- Generate QC message ---
def get_qc_message(sweep, df_sweeps):
    """Get error message"""
    if sweep not in df_sweeps["sweep_number"].values:
        return "<span style='color:red;'>Sweep number not found in the jsons!</span>"
    if sweep in df_sweeps.query("passed != passed")["sweep_number"].values:
        return "<span style='background:salmon;'>Sweep terminated by the experimenter!</span>"
    if sweep in df_sweeps.query("passed == False")["sweep_number"].values:
        return (
            f"<span style='background:yellow;'>Sweep failed QC! "
            f"({df_sweeps[df_sweeps.sweep_number == sweep].reasons.iloc[0][0]})</span>"
        )
    return "<span style='background:lightgreen;'>Sweep passed QC!</span>"


def pane_show_sweeps_of_one_cell(ephys_roi_id="1410790193"):
    if ephys_roi_id == "":
        return pn.pane.Markdown("Please select a cell from the table above.")

    # Load the NWB file.
    raw_this_cell = PatchSeqNWB(ephys_roi_id=ephys_roi_id)

    # Define a slider widget. Adjust the range based on your NWB data dimensions.
    slider = pn.widgets.IntSlider(
        name="Sweep number", start=0, end=raw_this_cell.n_sweeps - 1, value=0
    )

    # Bind the slider value to the update_plot function.
    plot_panel = pn.bind(update_plot, raw=raw_this_cell, sweep=slider.param.value)
    mpl_pane = pn.pane.Matplotlib(plot_panel, dpi=400, width=600, height=400)

    # Create a Tabulator widget for the DataFrame with row selection enabled.
    tab_sweeps = pn.widgets.Tabulator(
        raw_this_cell.df_sweeps[
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

    # Highlight rows based on the sweep metadata.
    tab_sweeps.style.apply(
        highlight_selected_rows,
        highlight_subset=raw_this_cell.df_sweeps.query("passed == True")["sweep_number"].tolist(),
        color="lightgreen",
        fields=["passed"],
        axis=1,
    ).apply(
        highlight_selected_rows,
        highlight_subset=raw_this_cell.df_sweeps.query("passed != passed")[
            "sweep_number"
        ].tolist(),  # NaN
        color="salmon",
        fields=["passed"],
        axis=1,
    ).apply(
        highlight_selected_rows,
        highlight_subset=raw_this_cell.df_sweeps.query("passed == False")["sweep_number"].tolist(),
        color="yellow",
        fields=["passed"],
        axis=1,
    ).apply(
        highlight_selected_rows,
        highlight_subset=raw_this_cell.df_sweeps.query("num_spikes > 0")["sweep_number"].tolist(),
        color="lightgreen",
        fields=["num_spikes"],
        axis=1,
    )

    # --- Two-Way Synchronization between Slider and Table ---
    # When the user selects a row in the table, update the slider.
    def update_slider_from_table(event):
        """table --> slider"""
        if event.new:
            # event.new is a list of selected row indices; assume single selection.
            selected_index = event.new[0]
            new_sweep = raw_this_cell.df_sweeps.loc[selected_index, "sweep_number"]
            slider.value = new_sweep

    tab_sweeps.param.watch(update_slider_from_table, "selection")

    # When the slider value changes, update the table selection.
    def update_table_selection(event):
        """Update slider --> table"""
        new_val = event.new
        row_index = raw_this_cell.df_sweeps.index[
            raw_this_cell.df_sweeps["sweep_number"] == new_val
        ].tolist()
        tab_sweeps.selection = row_index

    slider.param.watch(update_table_selection, "value")
    # --- End Synchronization ---

    sweep_msg = pn.bind(get_qc_message, sweep=slider.param.value, df_sweeps=raw_this_cell.df_sweeps)
    sweep_msg_panel = pn.pane.Markdown(sweep_msg, width=600, height=30)
    # --- End Error Message ---

    return pn.Row(
        pn.Column(
            pn.pane.Markdown(f"# {ephys_roi_id}"),
            pn.pane.Markdown("Use the slider to navigate through the sweeps in the NWB file."),
            pn.Column(slider, sweep_msg_panel, mpl_pane),
        ),
        pn.Column(
            pn.pane.Markdown("## Metadata from jsons"),
            tab_sweeps,
        ),
    )


# ---- Main Panel App Layout ----
def main():
    """main app"""

    # Create a Parameterized object to store the current ephys_roi_id.
    class DataHolder(param.Parameterized):
        ephys_roi_id = param.String(default="")

    data_holder = DataHolder()

    # ----

    pn.config.throttled = False

    df_meta = load_ephys_metadata()
    df_meta = df_meta.rename(
        columns={col: col.replace("_tab_master", "") for col in df_meta.columns}
    ).sort_values(["injection region"])

    cell_key = ["Date", "jem-id_cell_specimen", "ephys_roi_id", "ephys_qc", "injection region"]

    # MultiSelect widget to choose which columns to display.
    cols = list(df_meta.columns)
    cols.sort()
    col_selector = pn.widgets.MultiSelect(
        name="Add Columns to show",
        options=[col for col in cols if col not in cell_key],
        value=[],  # start with all columns
        height=500,
    )

    # Define a function to filter the DataFrame based on selected columns.
    def add_df_meta_col(selected_columns):
        return df_meta[cell_key + selected_columns]

    # Use pn.bind to create a reactive DataFrame that updates as the selection changes.
    filtered_df_meta = pn.bind(add_df_meta_col, col_selector)

    tab_df_meta = pn.widgets.Tabulator(
        filtered_df_meta,
        selectable=1,
        disabled=True,  # Not editable
        frozen_columns=cell_key,
        groupby=["injection region"],
        header_filters=True,
        show_index=False,
        height=500,
        width=1300,
        pagination=None,
        # page_size=15,
        stylesheets=[":host .tabulator {font-size: 12px;}"],
    )

    # When the user selects a row in the table, update the sweep view.
    def update_sweep_view_from_table(event):
        """table --> sweep view"""
        if event.new:
            # event.new is a list of selected row indices; assume single selection.
            selected_index = event.new[0]
            data_holder.ephys_roi_id = str(int(df_meta.iloc[selected_index]["ephys_roi_id"]))

    tab_df_meta.param.watch(update_sweep_view_from_table, "selection")

    pane_cell_selector = pn.Row(
        pn.Column(
            pn.pane.Markdown("## Cell selector"),
            pn.pane.Markdown(f"### Total LC-NE patch-seq cells: {len(df_meta)}"),
            width=400,
        ),
        col_selector,
        tab_df_meta,
    )

    # Layout
    pane_one_cell = pn.bind(
        pane_show_sweeps_of_one_cell, ephys_roi_id=data_holder.param.ephys_roi_id
    )

    layout = pn.Column(
        pn.pane.Markdown("# Patch-seq Ephys Data Navigator\n"),
        pane_cell_selector,
        pn.layout.Divider(),
        pane_one_cell,
    )

    # Make the panel servable if running with 'panel serve'
    return layout


layout = main()
layout.servable()
