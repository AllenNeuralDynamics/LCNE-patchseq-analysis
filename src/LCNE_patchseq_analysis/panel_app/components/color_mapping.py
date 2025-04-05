"""
Color mapping utilities for the scatter plot.
"""
from typing import Dict, Any, Union

import pandas as pd
from bokeh.models import CategoricalColorMapper, LinearColorMapper, ColorBar
from bokeh.plotting import figure

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER


class ColorMapping:
    """Handles color mapping for scatter plots."""

    def __init__(self, df_meta: pd.DataFrame):
        """Initialize with metadata dataframe."""
        self.df_meta = df_meta

    def add_color_bar(self, color_mapper: Union[CategoricalColorMapper, LinearColorMapper], 
                     title: str, p: figure) -> ColorBar:
        """Add a color bar to the plot with consistent styling."""
        color_bar = ColorBar(
            color_mapper=color_mapper,
            label_standoff=12,
            border_line_color=None,
            location=(0, 0),
            title=title,
            title_text_font_size="12pt",
            major_label_text_font_size="10pt",
        )
        p.add_layout(color_bar, "right")
        return color_bar

    def determine_color_mapping(self, color_mapping: str, color_palette: Any, 
                              p: figure) -> Dict[str, Any]:
        """
        Determine the color mapping for the scatter plot.
        
        Args:
            color_mapping: Column name to use for color mapping
            color_palette: Color palette to use
            p: Bokeh figure to add color bar to
            
        Returns:
            Dictionary with field and transform for scatter plot
        """
        if color_mapping == "injection region":
            color_mapper = {
                key: value
                for key, value in REGION_COLOR_MAPPER.items()
                if key in self.df_meta["injection region"].unique()
            }
            color_mapper = CategoricalColorMapper(
                factors=list(color_mapper.keys()), palette=list(color_mapper.values())
            )
            
            # Add a color bar for categorical data
            self.add_color_bar(color_mapper, color_mapping, p)
            
            return {"field": color_mapping, "transform": color_mapper}
            
        # If categorical (nunique <= 10), use categorical color mapper
        if self.df_meta[color_mapping].nunique() <= 10:
            color_mapper = CategoricalColorMapper(
                factors=list(self.df_meta[color_mapping].unique()),
                palette=color_palette[self.df_meta[color_mapping].nunique()]
            )
            self.add_color_bar(color_mapper, color_mapping, p)
            return {"field": color_mapping, "transform": color_mapper}
        
        # Try to convert the column to numeric
        numeric_data = pd.Series(pd.to_numeric(self.df_meta[color_mapping], errors="coerce"))
        if not numeric_data.isna().all():
            # If conversion is successful, use linear color mapper
            low = numeric_data.min()
            high = numeric_data.max()
            color_mapper = LinearColorMapper(palette=color_palette, low=low, high=high)
            color = {"field": color_mapping, "transform": color_mapper}
            
            # Add a color bar
            self.add_color_bar(color_mapper, color_mapping, p)
            return color
            
        return "black" 