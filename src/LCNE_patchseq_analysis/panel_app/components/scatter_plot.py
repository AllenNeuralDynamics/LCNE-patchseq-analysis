"""
Scatter plot component for the visualization app.
"""
import logging
from typing import List, Tuple, Any, Dict

import pandas as pd
import panel as pn
from bokeh.models import HoverTool, ColumnDataSource, BoxZoomTool, DatetimeTickFormatter
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import numpy as np

from LCNE_patchseq_analysis.panel_app.components.color_mapping import ColorMapping
from LCNE_patchseq_analysis.panel_app.components.size_mapping import SizeMapping
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_url_cell_summary

logger = logging.getLogger(__name__)


class ScatterPlot:
    """Handles scatter plot creation and updates."""

    def __init__(self, df_meta: pd.DataFrame, data_holder: Any):
        """Initialize with metadata dataframe."""
        self.df_meta = df_meta
        self.color_mapping = ColorMapping(df_meta)
        self.size_mapping = SizeMapping(df_meta)
        self.data_holder = data_holder
        # Add cell summary URLs to dataframe
        self._add_cell_summary_urls()

    def _add_cell_summary_urls(self):
        """Add cell summary URLs to the dataframe."""
        # Create a new column for cell summary URLs
        self.df_meta['cell_summary_url'] = None
        
        # Get URLs for each ephys_roi_id
        for idx, row in self.df_meta.iterrows():
            ephys_roi_id = str(int(row['ephys_roi_id']))
            try:
                url = get_public_url_cell_summary(ephys_roi_id, if_check_exists=False)
                self.df_meta.at[idx, 'cell_summary_url'] = url
            except Exception as e:
                logger.warning(f"Could not get URL for ephys_roi_id {ephys_roi_id}: {e}")
                self.df_meta.at[idx, 'cell_summary_url'] = None

    def create_plot_controls(self, width: int = 180) -> Dict[str, Any]:
        """Create the control widgets for the scatter plot."""
        # Get numeric and categorical columns
        numeric_cols = self.df_meta.select_dtypes(include=["number"]).columns.tolist()
        categorical_cols = self.df_meta.select_dtypes(include=["object"]).columns.tolist()
        all_cols = ["None"] + sorted(numeric_cols + categorical_cols)

        controls = {
            "x_axis_select": pn.widgets.Select(
                name="X Axis",
                options=all_cols,
                value="first_spike_AP_duration_half_width @ long_square_rheo, min",
                width=width,
            ),
            "y_axis_select": pn.widgets.Select(
                name="Y Axis",
                options=all_cols,
                value="Y (D --> V)",
                width=width,
            ),
            "color_col_select": pn.widgets.Select(
                name="Color By",
                options=all_cols,
                value="injection region",
                width=width,
            ),
            "size_col_select": pn.widgets.Select(
                name="Size By",
                options=all_cols,
                value="sag_ratio1 @ subthreshold, aver",
                width=width,
            ),
            "size_range_slider": pn.widgets.RangeSlider(
                name="Size Range",
                start=5,
                end=40,
                value=(10, 30),
                step=1,
                width=width,
            ),
            "size_gamma_slider": pn.widgets.FloatSlider(
                name="Size Gamma",
                start=0.1,
                end=5,
                value=1,
                step=0.1,
                width=width,
            ),
            "alpha_slider": pn.widgets.FloatSlider(
                name="Alpha",
                start=0.1,
                end=1,
                value=0.7,
                step=0.1,
                width=width,
            ),
            "width_slider": pn.widgets.IntSlider(
                name="Width",
                start=400,
                end=1200,
                value=800,
                step=50,
                width=width,
            ),
            "height_slider": pn.widgets.IntSlider(
                name="Height",
                start=400,
                end=1200,
                value=600,
                step=50,
                width=width,
            ),
            "bins_slider": pn.widgets.IntSlider(
                name="Histogram bins",
                start=10,
                end=100,
                value=50,
                step=1,
                width=width,
            ),
            "show_gmm": pn.widgets.Checkbox(
                name="Show Gaussian Mixture Model",
                value=True,
                width=width,
            ),
            "n_components_x": pn.widgets.IntSlider(
                name="Number of components (X)",
                start=1,
                end=5,
                value=2,
                step=1,
                width=width,
                disabled=False,
            ),
            "n_components_y": pn.widgets.IntSlider(
                name="Number of components (Y)",
                start=1,
                end=5,
                value=1,
                step=1,
                width=width,
                disabled=False,
            ),
            "hist_height_slider": pn.widgets.IntSlider(
                name="Distribution plot height",
                start=50,
                end=300,
                value=100,
                step=10,
            ),
        }
        
        # Link the GMM checkbox to enable/disable the component sliders
        def toggle_gmm_components(event):
            controls["n_components_x"].disabled = not event.new
            controls["n_components_y"].disabled = not event.new
            
        controls["show_gmm"].param.watch(toggle_gmm_components, "value")
        
        # Initialize the disabled state based on the initial checkbox value
        controls["n_components_x"].disabled = not controls["show_gmm"].value
        controls["n_components_y"].disabled = not controls["show_gmm"].value
        
        return controls

    def create_tooltips(
        self, x_col: str, y_col: str, color_col: str, size_col: str
    ) -> List[Tuple[str, str]]:
        """Create tooltips for the hover tool."""
        
        tooltips = f"""
             <div style="text-align: left; flex: auto; white-space: nowrap; margin: 0 10px">
                    <span style="font-size: 17px;">
                        <b>@Date_str, @{{injection region}}, @{{jem-id_cell_specimen}}, #@{{ephys_roi_id}}</b><br>
                        <b>X = @{{{x_col}}}</b> [{x_col}]<br>
                        <b>Y = @{{{y_col}}}</b> [{y_col}]<br>
                        <b> Color = @{{{color_col}}}</b> [{color_col}]<br>
                        <b> Size = @{{{size_col}}}</b> [{size_col}]<br>
                    </span>
             </div>
             <div>
                 <img src="@cell_summary_url{{safe}}" alt="Cell Summary" style="width: 800px; height: auto;">
             </div>
             """
            
        return tooltips

    def create_marginal_histogram(
        self, data: pd.Series, orientation: str, width: int, height: int, alpha: float, bins: int,
        show_gmm: bool = False, n_components: int = 1
    ) -> figure:
        """Create a histogram for marginal distribution with optional GMM overlay."""
        # Remove NaN values and convert to numeric
        clean_data = pd.to_numeric(data, errors='coerce').dropna()

        # If no valid data, create an empty plot
        if clean_data.empty:
            p = figure(
                height=height,
                width=width,
                tools="",
                toolbar_location=None,
                x_range=(0, 1),
                y_range=(0, 1),
            )
            p.text(
                x=0.5,
                y=0.5,
                text=["No valid data"],
                text_align="center",
                text_baseline="middle",
            )
            return p

        # Calculate histogram data (independent of orientation)
        hist, edges = np.histogram(clean_data, bins=bins, density=True)

        # Set axis ranges and quad parameters based on orientation
        if orientation == "x":
            x_range = (edges[0], edges[-1])
            y_range = (0, hist.max() * 1.1)
        else:  # "y" orientation
            x_range = (0, hist.max() * 1.1)
            y_range = (edges[0], edges[-1])

        # Create the figure
        p = figure(
            height=height,
            width=width,
            tools="",
            toolbar_location=None,
            x_range=x_range,
            y_range=y_range,
        )

        # Plot the histogram using vbar/hbar (Bokeh's bar plot) instead of quad
        if orientation == "x":
            # Use vbar for x-orientation
            p.vbar(x=[(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)],
                   top=hist,
                   width=(edges[1]-edges[0]) * 0.9,  # Slightly narrower than bin width
                   fill_color="gray",
                   line_color="white",
                   alpha=0.9)
        else:  # "y" orientation
            # Use hbar for y-orientation
            p.hbar(y=[(edges[i] + edges[i+1])/2 for i in range(len(edges)-1)],
                   right=hist,
                   height=(edges[1]-edges[0]) * 0.9,  # Slightly narrower than bin width
                   fill_color="gray",
                   line_color="white",
                   alpha=0.9)

        # Optional: Plot Gaussian Mixture Model overlay
        if show_gmm:
            from sklearn.mixture import GaussianMixture

            gmm = GaussianMixture(n_components=n_components, random_state=42)
            gmm.fit(clean_data.values.reshape(-1, 1))
            domain = np.linspace(edges[0], edges[-1], 1000)
            density = np.exp(gmm.score_samples(domain.reshape(-1, 1)))

            p.line(*((domain, density) if orientation == "x" else (density, domain)),
                   line_color="black", line_width=4, alpha=0.9)

            # Plot individual GMM components
            for i in range(n_components):
                mean = gmm.means_[i][0]
                std = np.sqrt(gmm.covariances_[i][0][0])
                weight = gmm.weights_[i]
                comp_density = weight * \
                    np.exp(-0.5 * ((domain - mean) / std) ** 2) / (std * np.sqrt(2 * np.pi))
                p.line(*((domain, comp_density) if orientation == "x" else (comp_density, domain)),
                       line_color="black", line_width=2, alpha=0.9, line_dash="dashed")

        # Hide axes and grid
        p.xaxis.visible = False
        p.yaxis.visible = False
        p.grid.visible = False
        return p


    def update_scatter_plot(
        self,
        x_col: str,
        y_col: str,
        color_col: str,
        color_palette: str,
        size_col: str,
        size_range: tuple,
        size_gamma: float,
        alpha: float,
        width: int,
        height: int,
        bins: int = 30,
        hist_height_slider: int = 100,
        show_gmm: bool = False,
        n_components_x: int = 2,
        n_components_y: int = 1,
    ) -> gridplot:
        """Update the scatter plot with new parameters."""
        # Create a new figure for the main scatter plot
        p = figure(
            x_axis_label=x_col,
            y_axis_label=y_col,
            tools="pan,wheel_zoom,box_zoom,reset,tap",
            height=height,
            width=width,
        )

        # Create ColumnDataSource from the dataframe
        source = ColumnDataSource(self.df_meta)
        
        # If any column is Date, convert it to datetime
        if x_col == "Date":
            source.data[x_col] = pd.to_datetime(
                pd.Series(source.data[x_col]), errors="coerce")
            p.xaxis.formatter = DatetimeTickFormatter(
                years="%Y",
                months="%Y-%m",
                days="%Y-%m-%d",
            )
            
        # Determine color mapping
        color = self.color_mapping.determine_color_mapping(color_col, color_palette, p)
        
        # Determine size mapping
        size = self.size_mapping.determine_size_mapping(
            size_col, source, min_size=size_range[0], max_size=size_range[1], gamma=size_gamma
        )

        # Add scatter glyph using the data source
        p.scatter(x=x_col, y=y_col, source=source, size=size, color=color, alpha=alpha)

        # Flip the y-axis if y_col is depth
        if y_col == "Y (D --> V)":
            p.y_range.flipped = True

        # Add HoverTool with tooltips
        tooltips = self.create_tooltips(x_col, y_col, color_col, size_col)
        hovertool = HoverTool(
            tooltips=tooltips,
            attachment="right",  # Fix tooltip to the right of the plot
            formatters={
                '@Date': "datetime"
            }
        )

        p.add_tools(hovertool)
                

        # Define callback to update ephys_roi_id on point tap
        def update_ephys_roi_id(attr, old, new):
            if new:
                selected_index = new[0]
                ephys_roi_id = str(int(self.df_meta.iloc[selected_index]["ephys_roi_id"]))
                logger.info(f"Selected ephys_roi_id: {ephys_roi_id}")
                # Update the data holder's ephys_roi_id
                if hasattr(self, "data_holder"):
                    self.data_holder.ephys_roi_id = ephys_roi_id

        # Attach the callback to the selection changes
        source.selected.on_change("indices", update_ephys_roi_id)
        
        # Set the default tool activated on drag to be box zoom
        p.toolbar.active_drag = p.select_one(BoxZoomTool)

        # Set axis label font sizes
        p.xaxis.axis_label_text_font_size = "14pt"
        p.yaxis.axis_label_text_font_size = "14pt"

        # Set major tick label font sizes
        p.xaxis.major_label_text_font_size = "12pt"
        p.yaxis.major_label_text_font_size = "12pt"

        # Create marginal histograms
        x_hist = None
        try:
            if x_col != "Date" and x_col != "None":  # Skip histogram for Date column
                x_hist = self.create_marginal_histogram(
                    self.df_meta[x_col], "x", width=width, height=hist_height_slider, alpha=alpha, bins=bins,
                    show_gmm=show_gmm, n_components=n_components_x
                )
                x_hist.x_range = p.x_range  # Link x ranges
        except Exception as e:
            logger.warning(f"Could not create x histogram: {e}")
            x_hist = None

        y_hist = None
        try:
            if y_col != "Date" and y_col != "None":  # Skip histogram for Date column
                y_hist = self.create_marginal_histogram(
                    self.df_meta[y_col], "y", width=hist_height_slider, height=height, alpha=alpha, bins=bins,
                    show_gmm=show_gmm, n_components=n_components_y
                )
                y_hist.y_range = p.y_range  # Link y ranges
        except Exception as e:
            logger.warning(f"Could not create y histogram: {e}")
            y_hist = None

        # Create grid layout
        layout = gridplot(
            [[y_hist, p], [None, x_hist]],
            toolbar_location="right",
            merge_tools=True,
            toolbar_options=dict(logo=None),
        )

        return layout 
