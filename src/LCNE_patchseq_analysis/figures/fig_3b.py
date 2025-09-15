import logging
from typing import Mapping, Sequence

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from LCNE_patchseq_analysis import REGION_COLOR_MAPPER
from LCNE_patchseq_analysis.figures import sort_region
from LCNE_patchseq_analysis.figures.util import save_figure

logger = logging.getLogger(__name__)


def generate_scatter_plot(
	df: pd.DataFrame,
	y_col: str,
	x_col: str,
	color_col: str,
	color_palette: Mapping[str, str] | None = None,
	plot_linear_regression: bool = True,
	point_size: int = 40,
	alpha: float = 0.8,
	figsize: tuple = (4, 4),
):
	"""Generic scatter plot utility for Figure 3B style.

	Args:
		df: Input dataframe.
		y_col: Column for y axis.
		x_col: Column for x axis.
		color_col: Column that defines groups / colors.
		color_palette: Mapping from group value to color. If None uses REGION_COLOR_MAPPER for injection regions.
		plot_linear_regression: If True overlay OLS line across all points and annotate p and R^2.
		point_size: Scatter marker size.
		alpha: Point transparency.
		figsize: Figure size.
	Returns:
		(fig, ax)
	"""
	if color_palette is None and color_col.lower() == "injection region":
		color_palette = REGION_COLOR_MAPPER

	# Drop rows missing required columns
	required_cols = [y_col, x_col, color_col]
	df_plot = df.dropna(subset=required_cols).copy()
	if df_plot.empty:
		raise ValueError("No data left after dropping NA for required columns")

	# Determine plotting order if injection region
	hue_order: Sequence[str] | None = None
	if color_col.lower() == "injection region":
		unique_regions = df_plot[color_col].dropna().unique().tolist()
		hue_order = sort_region(unique_regions)

	fig, ax = plt.subplots(figsize=figsize)
	sns.scatterplot(
		data=df_plot,
		x=x_col,
		y=y_col,
		hue=color_col,
		palette=color_palette,
		hue_order=hue_order,
		s=point_size,
		alpha=alpha,
		edgecolor="k",
		linewidth=0.3,
		ax=ax,
	)

	# Optional regression across all points (ignoring grouping)
	if plot_linear_regression:
		try:
			from scipy.stats import linregress  # type: ignore

			res = linregress(df_plot[x_col], df_plot[y_col])
			x_vals = pd.Series(sorted(df_plot[x_col].values))
			y_fit = res.intercept + res.slope * x_vals
			ax.plot(x_vals, y_fit, color="black", linewidth=1.2, zorder=5, label="Linear fit")
			# Annotation: p-value and R^2
			r_squared = res.rvalue ** 2
			annotation = f"p={res.pvalue:.2e}\nR^2={r_squared:.2f}"
			ax.text(
				0.98,
				0.02,
				annotation,
				transform=ax.transAxes,
				ha="right",
				va="bottom",
				fontsize=9,
				bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="gray", lw=0.5),
			)
		except Exception as e:  # noqa: BLE001 - lightweight handling, controlled scope
			logger.warning(f"Linear regression failed: {e}")

	ax.set_xlabel(x_col)
	ax.set_ylabel(y_col)
	ax.legend(title=color_col, loc="best", fontsize=8)
	sns.despine(ax=ax)
	fig.tight_layout()
	return fig, ax


def figure_3b_imputed_scRNAseq(
	df_meta: pd.DataFrame,
	filter_query: str | None = None,
	if_save_figure: bool = True,
	plot_linear_regression: bool = True,
):
	"""Figure 3B: Scatter of imputed scRNAseq pseudoclusters vs anatomical y coordinate.

	Assumptions:
		- x-axis coordinate column is named 'y' (consistent with figure 3a usage of anatomical y).
		- Imputed pseudocluster column name here assumed 'gene_imp_pseudoclusters (log_normed)'.
	"""
	if filter_query:
		df_meta = df_meta.query(filter_query)

	fig, ax = generate_scatter_plot(
		df=df_meta,
		y_col="gene_imp_pseudoclusters (log_normed)",
		x_col="y",
		color_col="injection region",
		color_palette=REGION_COLOR_MAPPER,
		plot_linear_regression=plot_linear_regression,
	)
	# Optional axis inversion: comment out unless required
	# ax.invert_xaxis()

	if if_save_figure:
		save_figure(
			fig,
			filename="fig_3b_scatter_imp_pseudoclusters_vs_y",
			dpi=300,
			formats=("png", "pdf"),
		)
	return fig, ax


def figure_3b_imputed_MERFISH(
	df_meta: pd.DataFrame,
	filter_query: str | None = None,
	if_save_figure: bool = True,
	plot_linear_regression: bool = True,
):
	"""Figure 3B: Scatter of imputed MERFISH pseudoclusters vs anatomical y coordinate.

	Assumptions:
		- x-axis coordinate column is named 'y' (consistent with figure 3a usage of anatomical y).
		- Imputed pseudocluster column name here assumed 'gene_imp_pseudoclusters (log_normed)'.
	"""
	if filter_query:
		df_meta = df_meta.query(filter_query)

	fig, ax = generate_scatter_plot(
		df=df_meta,
		y_col="gene_imp_DV (log_normed)",
		x_col="y",
		color_col="injection region",
		color_palette=REGION_COLOR_MAPPER,
		plot_linear_regression=plot_linear_regression,
	)
	# Optional axis inversion: comment out unless required
	# ax.invert_xaxis()

	if if_save_figure:
		save_figure(
			fig,
			filename="fig_3b_scatter_imp_MERFISH_vs_y",
			dpi=300,
			formats=("png", "pdf"),
		)
	return fig, ax

if __name__ == "__main__":
	from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
	from LCNE_patchseq_analysis.figures import GLOBAL_FILTER

	df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)

    # For gene data, apply additional filtering
	gene_filter = GLOBAL_FILTER + " & mapmycells_subclass_name.str.contains('DBH', case=False, na=False)"
	figure_3b_imputed_scRNAseq(df_meta, gene_filter)
	figure_3b_imputed_MERFISH(df_meta, gene_filter)

