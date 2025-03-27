"""Lightweight panel app for navigating NWB files.
Run this in command line:
    panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org
"""

import matplotlib.pyplot as plt
import numpy as np
import panel as pn
import pandas as pd

from bokeh.models.widgets.tables import NumberFormatter, BooleanFormatter

from LCNE_patchseq_analysis import RAW_DIRECTORY
from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB

def load_ephys_metadata():
    """Load ephys metadata
    
    Per discussion with Brian, we should only look at those in the spreadsheet.
    https://www.notion.so/hanhou/LCNE-patch-seq-analysis-1ae3ef97e735808eb12ec452d2dc4369?pvs=4#1ba3ef97e73580ac9a5ee6e53e9b3dbe  # noqa: E501
    """
    df = pd.read_csv(RAW_DIRECTORY + "/df_metadata_merged.csv")
    df = df.query("spreadsheet_or_lims in ('both', 'spreadsheet_only')")

    # Rename "Crus 1" to "Crus1"
    df.loc[
        df["injection region"].astype(str).str.contains("Crus", na=False),
        "injection region",
    ] = "Crus 1"
    return df

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
    style = [''] * len(row)
    if row['sweep_number'] in highlight_subset:
        if fields is None:
            return [f'background-color: {color}'] * len(row)
        else:
            for field in fields:
                style[list(row.keys()).index(field)] = f'background-color: {color}'
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


def panel_show_sweeps_of_one_cell(ephys_roi_id="1410790193"):
    # Load the NWB file.
    raw_this_cell = PatchSeqNWB(ephys_roi_id=ephys_roi_id)

    # Define a slider widget. Adjust the range based on your NWB data dimensions.
    slider = pn.widgets.IntSlider(name="Sweep number", start=0, end=raw_this_cell.n_sweeps - 1, value=0)

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
        highlight_subset=raw_this_cell.df_sweeps.query("passed != passed")["sweep_number"].tolist(), # NaN
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
        row_index = raw_this_cell.df_sweeps.index[raw_this_cell.df_sweeps["sweep_number"] == new_val].tolist()
        tab_sweeps.selection = row_index

    slider.param.watch(update_table_selection, "value")
    # --- End Synchronization ---

    sweep_msg = pn.bind(get_qc_message, sweep=slider.param.value, df_sweeps=raw_this_cell.df_sweeps)
    sweep_msg_panel = pn.pane.Markdown(sweep_msg, width=600, height=30)
    # --- End Error Message ---

    return pn.Row(
        pn.Column(
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

    pn.config.throttled = False

    df_meta = load_ephys_metadata()
    df_meta = df_meta.rename(
        columns={col: col.replace("_tab_master", "") for col in df_meta.columns}
    ).sort_values(["injection region"])

    bokeh_formatters = {
        'float': NumberFormatter(format='0.0000'),
        'bool': BooleanFormatter(),
        'int': NumberFormatter(format='0'),
    }

    tab_df_meta = pn.widgets.Tabulator(
        df_meta,
        selectable=1,
        disabled=True,  # Not editable
        frozen_columns=["Date", "jem-id_cell_specimen", "ephys_roi_id", "ephys_qc"],
        groupby=["injection region"],
        header_filters=True,
        show_index=False,
        height=500,
        width=1300,
        pagination=None,
        # page_size=15,
        stylesheets=[":host .tabulator {font-size: 12px;}"],
        formatters=bokeh_formatters,
    )

    pane_cell_selector = pn.Row(
        pn.Column(
            pn.pane.Markdown("## Cell selector"),
            pn.pane.Markdown(f"### Total LC-NE patch-seq cells: {len(df_meta)}"),
            width=400
        ),
        tab_df_meta,
    )

    # Layout
    pane_one_cell = panel_show_sweeps_of_one_cell(
        ephys_roi_id="1417382638"
    )
    layout = pn.Column(
        pn.pane.Markdown("# Patch-seq Ephys Data Navigator\n"),
        pane_cell_selector,
        pane_one_cell,
    )

    # Make the panel servable if running with 'panel serve'
    return layout


layout = main()
layout.servable()
