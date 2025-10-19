import matplotlib.pyplot as plt

from LCNE_patchseq_analysis.figures.fig_3a import figure_3a_ccf_sagittal
from LCNE_patchseq_analysis.figures.fig_3b import (
    figure_3b_imputed_MERFISH,
    figure_3b_imputed_scRNAseq,
)
from LCNE_patchseq_analysis.figures.fig_3c import (
    figure_3c_latency_comparison,
    figure_3c_tau_comparison,
)
from LCNE_patchseq_analysis.figures.util import save_figure


def generate_main_figure(
    df_meta,
    global_filter: str,
    gene_filter: str,
    if_save_figure: bool = True,
):
    # --- Layout ---
    fig = plt.figure(constrained_layout=False, figsize=(10, 13))
    gs0 = fig.add_gridspec(3, 1, height_ratios=[1.3, 1, 1], width_ratios=[1], hspace=0.5)

    gs0_0 = gs0[0].subgridspec(1, 2, width_ratios=[1.5, 1], wspace=0.3)
    gs0_1 = gs0[1].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.5)
    gs0_2 = gs0[2].subgridspec(1, 2, width_ratios=[1, 1], wspace=0.5)

    ax0_0 = fig.add_subplot(gs0_0[0, 0])  # 3a left
    # ax0_1 = fig.add_subplot(gs0_0[0, 1])  # 3a right
    ax1_0 = fig.add_subplot(gs0_1[0, 0])  # 3b left
    ax1_1 = fig.add_subplot(gs0_1[0, 1])  # 3b right
    ax2_0 = fig.add_subplot(gs0_2[0, 0])  # 3c left
    ax2_1 = fig.add_subplot(gs0_2[0, 1])  # 3c right

    figure_3a_ccf_sagittal(df_meta, global_filter, ax=ax0_0, if_save_figure=False)
    ax0_0.set_title("")
    ax0_0.get_legend().remove()

    # figure_3a_ycoord_violin(df_meta, global_filter, ax=ax0_1, if_save_figure=False)
    # ax0_1.set_position([0.55, 0.68, 0.2, 0.15])  # [left, bottom, width, height]

    _, ax1_0 = figure_3b_imputed_scRNAseq(df_meta, gene_filter, ax=ax1_0, if_save_figure=False)
    ax1_0.get_legend().remove()
    _, ax1_1 = figure_3b_imputed_MERFISH(df_meta, gene_filter, ax=ax1_1, if_save_figure=False)
    ax1_1.get_legend().remove()

    _, ax2_0 = figure_3c_tau_comparison(df_meta, global_filter, ax=ax2_0, if_save_figure=False)
    ax2_0.get_legend().remove()
    _, ax2_1 = figure_3c_latency_comparison(df_meta, global_filter, ax=ax2_1, if_save_figure=False)
    ax2_1.get_legend().remove()

    if if_save_figure:
        save_figure(
            fig, filename="main_figure", dpi=300, formats=("png", "pdf"), bbox_inches="tight"
        )
        print("Figure saved as main_figure.png/.pdf")
    return fig


if __name__ == "__main__":

    from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata

    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    from LCNE_patchseq_analysis.figures import GENE_FILTER, GLOBAL_FILTER

    generate_main_figure(df_meta, GLOBAL_FILTER, GENE_FILTER)
