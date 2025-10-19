"""Aggregated access to supplemental figure generators."""

from LCNE_patchseq_analysis.figures.fig_3a import sup_figure_3a_ccf_coronal  # noqa: F401
from LCNE_patchseq_analysis.figures.fig_3c import (  # noqa: F401
    sup_figure_3b_all_gene_features,
    sup_figure_3c_all_ipfx_features,
    sup_figure_3d_morphology,
)

if __name__ == "__main__":  # Simple manual smoke test
    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
    from LCNE_patchseq_analysis.figures import GENE_FILTER, GLOBAL_FILTER

    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True, if_with_morphology=True)
    sup_figure_3a_ccf_coronal(df_meta, GLOBAL_FILTER)

    sup_figure_3c_all_ipfx_features(df_meta, GLOBAL_FILTER)
    sup_figure_3b_all_gene_features(df_meta, GENE_FILTER)
    sup_figure_3d_morphology(df_meta, GLOBAL_FILTER)
