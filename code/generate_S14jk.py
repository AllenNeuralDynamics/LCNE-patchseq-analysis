"""Reproduce supplementary figure S14 panels j and k side by side.

Panel j: raw example voltage traces for one example cell per projection target.
Panel k: PC1 and membrane time constant cumulative distributions.
Colors are shared between the two panels.
"""

import logging

from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes

from LCNE_patchseq_analysis.figures import GLOBAL_FILTER
from LCNE_patchseq_analysis.figures.main_pca_tau import figure_s14_jk

# -- Physiological properties --
logger = logging.getLogger(__name__)

logging.basicConfig(level=logging.INFO)
logger.info("Loading metadata...")

df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
df_meta = df_meta.query(GLOBAL_FILTER)
logger.info(f"Loaded metadata with shape: {df_meta.shape}")

logger.info("Loading spike waveforms...")
df_spikes = get_public_representative_spikes("average")

logger.info("Generating figure S14 panels j (example traces) and k (cumulative distributions)...")
fig, axes_dict, results, csv_data = figure_s14_jk(
    df_meta, df_spikes, filtered_df_meta=df_meta
)
