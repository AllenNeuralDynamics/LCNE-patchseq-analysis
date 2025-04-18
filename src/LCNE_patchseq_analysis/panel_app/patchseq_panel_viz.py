"""
Panel-based visualization tool for navigating and visualizing patch-seq NWB files.

To start the app, run:
panel serve panel_nwb_viz.py --dev --allow-websocket-origin=codeocean.allenneuraldynamics.org --title "Patch-seq Data Explorer"  # noqa: E501
"""

import logging

import pandas as pd
import panel as pn
import param
from bokeh.io import curdoc
from bokeh.layouts import column as bokeh_column
from bokeh.models import (
    BoxZoomTool,
)
from bokeh.plotting import figure

from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.data_util.nwb import PatchSeqNWB
from LCNE_patchseq_analysis.efel.io import load_efel_features_from_roi
from LCNE_patchseq_analysis.panel_app.components.scatter_plot import ScatterPlot
from LCNE_patchseq_analysis.panel_app.components.spike_analysis import RawSpikeAnalysis
from LCNE_patchseq_analysis.pipeline_util.s3 import (
    get_public_url_cell_summary,
    get_public_url_sweep,
    S3_PUBLIC_URL_BASE,
)

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# pn.extension("tabulator")
curdoc().title = "LC-NE Patch-seq Data Explorer"


class PatchSeqNWBApp(param.Parameterized):
    """
    Object-Oriented Panel App for navigating NWB files.
    Encapsulates metadata loading, sweep visualization, and cell selection.
    """

    class DataHolder(param.Parameterized):
        """
        Holder for currently selected cell ID and sweep number.
        """

        ephys_roi_id_selected = param.String(default="")
        sweep_number_selected = param.Integer(default=0)

    def __init__(self):
        """
        Initialize the PatchSeqNWBApp.
        """
        # Holder for currently selected cell ID.
        self.data_holder = PatchSeqNWBApp.DataHolder()

        # Load and prepare metadata.
        self.df_meta = load_ephys_metadata(if_with_efel=True)
        self.df_meta.rename(
            columns={
                "x": "X (A --> P)",
                "y": "Y (D --> V)",
                "z": "Z (L --> R)",
            },
            inplace=True,
        )

        self.cell_key = [
            "Date",
            "jem-id_cell_specimen",
            "ephys_roi_id",
            "ephys_qc",
            "LC_targeting",
            "injection region",
            "Y (D --> V)",
        ]
        # Turn Date to datetime
        self.df_meta.loc[:, "Date_str"] = self.df_meta["Date"]  # Keep the original Date as string
        self.df_meta.loc[:, "Date"] = pd.to_datetime(self.df_meta["Date"], errors="coerce")

        # Initialize scatter plot component
        self.scatter_plot = ScatterPlot(self.df_meta, self.data_holder)
        
        # Initialize spike analysis component
        self.raw_spike_analysis = RawSpikeAnalysis(self.df_meta, main_app=self)

        # Create the cell selector panel once.
        self.cell_selector_panel = self.create_cell_selector_panel()

    @staticmethod
    def update_bokeh(raw, sweep, downsample_factor=3):
        """
        Update the Bokeh plot for a given sweep.
        """
        trace = raw.get_raw_trace(sweep)[::downsample_factor]
        stimulus = raw.get_stimulus(sweep)[::downsample_factor]
        time = raw.get_time(sweep)[::downsample_factor]

        box_zoom_auto = BoxZoomTool(dimensions="auto")

        # Create the voltage trace plot
        voltage_plot = figure(
            title=f"Full traces - Sweep number {sweep} (downsampled {downsample_factor}x)",
            height=300,
            tools=["hover", box_zoom_auto, "box_zoom", "wheel_zoom", "reset", "pan"],
            active_drag=box_zoom_auto,
            x_range=(0, time[-1]),
            y_axis_label="Vm (mV)",
            sizing_mode="stretch_width",
        )
        voltage_plot.line(time, trace, line_width=1.5, color="navy")

        # Create the stimulus plot
        stim_plot = figure(
            height=150,
            tools=["hover", box_zoom_auto, "box_zoom", "wheel_zoom", "reset", "pan"],
            active_drag=box_zoom_auto,
            x_range=voltage_plot.x_range,  # Link x ranges
            x_axis_label="Time (ms)",
            y_axis_label="I (pA)",
            sizing_mode="stretch_width",
        )
        stim_plot.line(time, stimulus, line_width=1.5, color="firebrick")

        # Stack the plots vertically using bokeh's column layout
        layout = bokeh_column(
            voltage_plot, stim_plot, sizing_mode="stretch_width", margin=(50, 0, 0, 0)
        )
        return layout

    @staticmethod
    def highlight_selected_rows(row, highlight_subset, color, fields=None):
        """
        Highlight rows based on a subset of values.
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

    @staticmethod
    def get_qc_message(sweep, df_sweeps):
        """Return a QC message based on sweep data."""
        if sweep not in df_sweeps["sweep_number"].values:
            return "<span style='color:red;'>Invalid sweep!</span>"
        if sweep in df_sweeps.query("passed != passed")["sweep_number"].values:
            return "<span style='background:salmon;'>Sweep terminated by the experimenter!</span>"
        if sweep in df_sweeps.query("passed == False")["sweep_number"].values:
            return (
                f"<span style='background:yellow;'>Sweep failed QC! "
                f"({df_sweeps[df_sweeps.sweep_number == sweep].reasons.iloc[0][0]})</span>"
            )
        return "<span style='background:lightgreen;'>Sweep passed QC!</span>"

    def create_scatter_plot(self):
        """
        Create the scatter plot panel using the ScatterPlot component.
        """
        control_width = 300
        
        # Get plot controls from the scatter plot component
        controls = self.scatter_plot.create_plot_controls(width=control_width)

        # Create a reactive scatter plot that updates when controls change
        scatter_plot = pn.bind(
            self.scatter_plot.update_scatter_plot,
            controls["x_axis_select"].param.value,
            controls["y_axis_select"].param.value,
            controls["color_col_select"].param.value,
            controls["color_palette_select"].param.value,
            controls["size_col_select"].param.value,
            controls["size_range_slider"].param.value_throttled,
            controls["size_gamma_slider"].param.value_throttled,
            controls["alpha_slider"].param.value_throttled,
            controls["width_slider"].param.value_throttled,
            controls["height_slider"].param.value_throttled,
            controls["font_size_slider"].param.value_throttled,
            controls["bins_slider"].param.value_throttled,
            controls["hist_height_slider"].param.value_throttled,
            controls["show_gmm"].param.value,
            controls["n_components_x"].param.value,
            controls["n_components_y"].param.value,
        )

        return pn.Row(
            pn.Column(
                controls["x_axis_select"],
                controls["y_axis_select"],
                pn.layout.Divider(margin=(5, 0, 5, 0)),
                controls["color_col_select"],
                controls["color_palette_select"],
                pn.layout.Divider(margin=(5, 0, 5, 0)),
                controls["size_col_select"],
                controls["size_range_slider"],
                controls["size_gamma_slider"],
                pn.layout.Divider(margin=(5, 0, 5, 0)),
                controls["bins_slider"],
                controls["show_gmm"],
                controls["n_components_x"],
                controls["n_components_y"],
                pn.layout.Divider(margin=(5, 0, 5, 0)),
                pn.Accordion(
                    (
                        "Plot settings",
                        pn.Column(
                            controls["alpha_slider"],
                            controls["width_slider"],
                            controls["height_slider"],
                            controls["hist_height_slider"],
                            controls["font_size_slider"],
                            width=control_width - 30,
                        ),
                    ),
                    active=[1],
                ),
                margin=(0, 50, 20, 0),  # top, right, bottom, left margins in pixels
                width=control_width,
            ),
            scatter_plot,
            margin=(0, 20, 20, 20),  # top, right, bottom, left margins in pixels
            # width=800,
        )

    def create_cell_selector_panel(self):
        """
        Builds and returns the cell selector panel that displays metadata.
        """
        # MultiSelect widget to choose additional columns.
        cols = list(self.df_meta.columns)
        cols.sort()
        selectable_cols = [col for col in cols if col not in self.cell_key]
        col_selector = pn.widgets.MultiSelect(
            name="Add more columns to show in the table",
            options=selectable_cols,
            value=[
                "width_rheo",
                "efel_AP_width @ long_square_rheo, aver",
                "sag",
                "efel_sag_ratio1 @ subthreshold, aver",
            ],  # start with no additional columns
            height=300,
            width=500,
        )

        def add_df_meta_col(selected_columns):
            return self.df_meta[self.cell_key + selected_columns]

        filtered_df_meta = pn.bind(add_df_meta_col, col_selector)
        tab_df_meta = pn.widgets.Tabulator(
            filtered_df_meta,
            selectable=1,
            disabled=True,  # Not editable
            frozen_columns=self.cell_key,
            groupby=["injection region"],
            header_filters=True,
            show_index=False,
            height=300,
            sizing_mode="stretch_width",
            pagination=None,
            stylesheets=[":host .tabulator {font-size: 12px;}"],
        )

        # When a row is selected, update the current cell (ephys_roi_id).
        def update_sweep_view_from_table(event):
            if event.new:
                selected_index = event.new[0]
                self.data_holder.ephys_roi_id_selected = str(
                    int(self.df_meta.iloc[selected_index]["ephys_roi_id"])
                )

        tab_df_meta.param.watch(update_sweep_view_from_table, "selection")

        scatter_plot = self.create_scatter_plot()

        # Add cell-level summary plot
        def get_s3_cell_summary_plot(ephys_roi_id):
            s3_url = get_public_url_cell_summary(ephys_roi_id)
            if s3_url:
                return pn.pane.PNG(s3_url, sizing_mode="stretch_width")
            else:
                return pn.pane.Markdown(
                    "### Select the table or the scatter plot to view the cell summary plot."
                )

        s3_cell_summary_plot = pn.Column(
            pn.bind(
                lambda ephys_roi_id: pn.pane.Markdown(
                    "## Cell summary plot" + (f" for {ephys_roi_id}" if ephys_roi_id else "")
                ),
                ephys_roi_id=self.data_holder.param.ephys_roi_id_selected,
            ),
            pn.bind(get_s3_cell_summary_plot, ephys_roi_id=self.data_holder.param.ephys_roi_id_selected),
            sizing_mode="stretch_width",
        )

        cell_selector_panel = pn.Column(
            pn.Row(
                col_selector,
                tab_df_meta,
                height=350,
            ),
            pn.Row(
                scatter_plot,
                s3_cell_summary_plot,
            ),
        )
        return cell_selector_panel

    def create_sweep_panel(self, ephys_roi_id=""):
        """
        Builds and returns the sweep visualization panel for a single cell.
        """
        if ephys_roi_id == "":
            return pn.pane.Markdown("Please select a cell from the table above.")

        # Load the NWB file for the selected cell.
        raw_this_cell = PatchSeqNWB(ephys_roi_id=ephys_roi_id, if_load_metadata=False)

        # Now let's get df sweep from the eFEL enriched one
        df_sweeps = load_efel_features_from_roi(ephys_roi_id, if_from_s3=True)["df_sweeps"]
        df_sweeps_valid = df_sweeps.query("passed == passed")

        # Set initial sweep number to first valid sweep
        if self.data_holder.sweep_number_selected == 0:
            self.data_holder.sweep_number_selected = df_sweeps_valid.iloc[0]["sweep_number"]

        # Add a slider to control the downsample factor
        downsample_factor = pn.widgets.IntSlider(
            name="Downsample factor",
            value=5,
            start=1,
            end=10,
        )

        # Bind the plotting function to the data holder's sweep number
        bokeh_panel = pn.bind(
            PatchSeqNWBApp.update_bokeh,
            raw=raw_this_cell,
            sweep=self.data_holder.param.sweep_number_selected,
            downsample_factor=downsample_factor.param.value_throttled,
        )

        # Bind the S3 URL retrieval to the data holder's sweep number
        def get_s3_sweep_images(sweep_number):
            s3_url = get_public_url_sweep(ephys_roi_id, sweep_number)
            images = []
            if isinstance(s3_url, dict) and "sweep" in s3_url:
                images.append(pn.pane.PNG(s3_url["sweep"], width=800, height=400))
            if isinstance(s3_url, dict) and "spikes" in s3_url:
                images.append(pn.pane.PNG(s3_url["spikes"], width=800, height=400))
            return pn.Column(*images) if images else pn.pane.Markdown("No S3 images available")

        s3_sweep_images_panel = pn.bind(
            get_s3_sweep_images, sweep_number=self.data_holder.param.sweep_number_selected
        )
        sweep_pane = pn.Column(
            s3_sweep_images_panel,
            bokeh_panel,
            downsample_factor,
            sizing_mode="stretch_width",
        )

        # Build a Tabulator for sweep metadata.
        tab_sweeps = pn.widgets.Tabulator(
            df_sweeps_valid[
                [
                    "sweep_number",
                    "stimulus_code_ext",
                    "stimulus_name",
                    "stimulus_amplitude",
                    "passed",
                    "efel_num_spikes",
                    "num_spikes",
                    "stimulus_start_time",
                    "stimulus_duration",
                    "tags",
                    "reasons",
                    "stimulus_code",
                ]
            ],  # Only show valid sweeps (passed is not NaN)
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

        # Apply conditional row highlighting.
        if hasattr(tab_sweeps, "style"):
            tab_sweeps.style.apply(
                PatchSeqNWBApp.highlight_selected_rows,
                highlight_subset=df_sweeps_valid.query("passed == True")["sweep_number"].tolist(),
                color="lightgreen",
                fields=["passed"],
                axis=1,
            ).apply(
                PatchSeqNWBApp.highlight_selected_rows,
                highlight_subset=df_sweeps_valid.query("passed != passed")["sweep_number"].tolist(),
                color="salmon",
                fields=["passed"],
                axis=1,
            ).apply(
                PatchSeqNWBApp.highlight_selected_rows,
                highlight_subset=df_sweeps_valid.query("passed == False")["sweep_number"].tolist(),
                color="yellow",
                fields=["passed"],
                axis=1,
            ).apply(
                PatchSeqNWBApp.highlight_selected_rows,
                highlight_subset=df_sweeps_valid.query("num_spikes > 0")["sweep_number"].tolist(),
                color="lightgreen",
                fields=["num_spikes"],
                axis=1,
            )

        # --- Synchronize table selection with sweep number ---
        def update_sweep_from_table(event):
            """Update sweep number when table selection changes."""
            if event.new:
                selected_index = event.new[0]
                new_sweep = df_sweeps_valid.iloc[selected_index]["sweep_number"]
                self.data_holder.sweep_number_selected = new_sweep

        tab_sweeps.param.watch(update_sweep_from_table, "selection")
        # --- End Synchronization ---

        # Build a reactive QC message panel.
        sweep_msg = pn.bind(
            PatchSeqNWBApp.get_qc_message,
            sweep=self.data_holder.param.sweep_number_selected,
            df_sweeps=df_sweeps,
        )
        sweep_msg_panel = pn.pane.Markdown(sweep_msg, width=600, height=30)

        return pn.Row(
            pn.Column(
                pn.pane.Markdown(f"# {ephys_roi_id}"),
                pn.pane.Markdown("Select a sweep from the table to view its data."),
                pn.Column(sweep_msg_panel, sweep_pane),
                width=700,
                margin=(0, 100, 0, 0),  # top, right, bottom, left margins
            ),
            pn.Column(
                pn.pane.Markdown("## Sweep metadata"),
                tab_sweeps,
            ),
        )

    def main_layout(self):
        """
        Constructs the full application layout.
        """
        pn.config.throttled = False
        pane_cell_selector = self.cell_selector_panel

        # Create spike analysis controls and plots
        spike_controls = self.raw_spike_analysis.create_plot_controls()

        def update_spike_plots(extract_from, n_clusters, alpha, width, height, 
                               marker_size, normalize_window_v, normalize_window_dvdt, 
                               if_show_cluster_on_retro, spike_range, dim_reduction_method):
            # Extract representative spikes
            df_v_norm, df_dvdt_norm = self.raw_spike_analysis.extract_representative_spikes(
                extract_from=extract_from,
                if_normalize_v=True,
                normalize_window_v=normalize_window_v,
                if_normalize_dvdt=True,
                normalize_window_dvdt=normalize_window_dvdt,
                if_smooth_dvdt=False,
            )
            
            # Create spike analysis plots
            return self.raw_spike_analysis.create_raw_PCA_plots(
                df_v_norm=df_v_norm,
                df_dvdt_norm=df_dvdt_norm,
                n_clusters=n_clusters,
                alpha=alpha,
                width=width,
                height=height,
                marker_size=marker_size,
                if_show_cluster_on_retro=if_show_cluster_on_retro,
                spike_range=spike_range,
                dim_reduction_method=dim_reduction_method,
            )
        
        # Create spike analysis plots
        controls = spike_controls  # shorter name for readability
        param_keys = [
            "n_clusters", "alpha_slider", "plot_width", "if_show_cluster_on_retro",
            "plot_height", "marker_size", "normalize_window_v", "normalize_window_dvdt",
            "spike_range", "dim_reduction_method"
        ]
        params = {
            k: controls[k].param.value_throttled
            if k not in ["if_show_cluster_on_retro", "dim_reduction_method"]
            else controls[k].param.value
            for k in param_keys
        }
        
        spike_plots = pn.bind(
            update_spike_plots,
            extract_from=controls["extract_from"].param.value,
            n_clusters=params["n_clusters"],
            alpha=params["alpha_slider"],
            width=params["plot_width"],
            height=params["plot_height"],
            marker_size=params["marker_size"],
            if_show_cluster_on_retro=params["if_show_cluster_on_retro"],
            normalize_window_v=params["normalize_window_v"],
            normalize_window_dvdt=params["normalize_window_dvdt"],
            spike_range=params["spike_range"],
            dim_reduction_method=params["dim_reduction_method"],
        )

        # Bind the sweep panel to the current cell selection.
        pane_one_cell = pn.bind(
            self.create_sweep_panel, ephys_roi_id=self.data_holder.param.ephys_roi_id_selected
        )

        # Create a toggle button for showing/hiding raw sweeps
        show_sweeps_button = pn.widgets.Button(name="Show raw sweeps", button_type="primary")
        show_sweeps = pn.widgets.Toggle(name="Show raw sweeps", value=False)

        # Link the button to the toggle
        def toggle_sweeps(event):
            show_sweeps.value = not show_sweeps.value
            show_sweeps_button.name = "Hide raw sweeps" if show_sweeps.value else "Show raw sweeps"

        show_sweeps_button.on_click(toggle_sweeps)

        # Create a dynamic layout that includes pane_one_cell only when show_sweeps is True
        dynamic_content = pn.bind(
            lambda show: pn.Column(pane_one_cell) if show else pn.Column(), show_sweeps.param.value
        )

        layout = pn.Column(
            pn.pane.Markdown("# Patch-seq Ephys Data Explorer\n"),

            pn.Column(
                pn.pane.Markdown(f"## Extracted Features (N = {len(self.df_meta)})"),
            ),
            pane_cell_selector,
            pn.layout.Divider(),            
            pn.Column(
                pn.pane.Markdown("## Raw Spikes"),
                pn.Row(
                    pn.Column(*spike_controls.values(), width=250),
                    spike_plots,
                ),
            ),
            pn.layout.Divider(),
            show_sweeps_button,
            dynamic_content,
            pn.layout.Divider(),
            pn.Accordion(
                (
                    "Distribution of all features",
                    pn.pane.PNG(S3_PUBLIC_URL_BASE + "/efel/cell_stats/distribution_all_features.png", 
                                width=1300),
                ),
            ),
            margin=(20, 20, 0, 20),  # top, right, bottom, left margins in pixels
        )
        return layout


app = PatchSeqNWBApp()
layout = app.main_layout()
layout.servable()
