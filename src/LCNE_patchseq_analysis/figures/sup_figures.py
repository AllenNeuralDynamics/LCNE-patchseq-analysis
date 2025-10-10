"""Aggregated access to supplemental figure generators.
"""

from LCNE_patchseq_analysis.figures.fig_3a import sup_figure_3a_ccf_coronal  # noqa: F401



if __name__ == "__main__":  # Simple manual smoke test
	from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
	from LCNE_patchseq_analysis.figures import GLOBAL_FILTER

	df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
	sup_figure_3a_ccf_coronal(df_meta, GLOBAL_FILTER)
