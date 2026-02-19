"""
Utilities for processing sequencing data.
"""

import logging
import os
from pathlib import Path

import pandas as pd

from LCNE_patchseq_analysis import PACKAGE_DIRECTORY

logger = logging.getLogger(__name__)

# Define the preselected columns of interest
SEQ_COLUMNS = [
    # --- New columns added in 0620 ---
    "RNA_QC",  # RNA quality control score
    "imp_pscores",  # Imputed pseudocluster scores
    # --- Imputation to other experiments ---
    "imp_ML",
    "imp_DV",
    "imp_AP",
    "imp_pseudoclusters",
    # --- Updated imputation from 02122026 (hyperparameter: gaussian_sigma0p1_k100) ---
    "imp_ML_gaussian_sigma0p1_k100",
    "imp_DV_gaussian_sigma0p1_k100",
    "imp_AP_gaussian_sigma0p1_k100",
    # --- Pseudocluster columns from 02122026 ---
    "imp_pseudoclusters_gaussian_sigma0p1_k100",
    "imp_pseudoclusters_gaussian_sigma1_k100",
    "imp_pseudoclusters_softmax_tau0p1_k100",
    "imp_pseudoclusters_softmax_tau0p5_k100",
    # --- Noradrenergic markers ---
    "Dbh",  # Dopamine beta-hydroxylase
    "Th",  # Tyrosine hydroxylase
    "Slc18a2",  # Vesicular monoamine transporter 2 (VMAT2)
    "Slc6a2",  # Norepinephrine transporter (NET)
    # --- Active ion channels ---
    # Sodium channels (important for action potential initiation and upstroke)
    "Scn1a",  # Nav1.1, important for AP initiation
    "Scn2a",  # Nav1.2, expressed in axon initial segment
    "Scn8a",  # Nav1.6, responsible for AP initiation at AIS
    "Scn9a",  # Nav1.7, influences AP threshold
    # Potassium channels (important for repolarization and spike width)
    "Kcna1",  # Kv1.1, delayed rectifier, affects spike width
    "Kcna2",  # Kv1.2, delayed rectifier, affects spike width
    "Kcnc1",  # Kv3.1, fast-activating, critical for narrow spikes
    "Kcnc2",  # Kv3.2, fast-activating, critical for narrow spikes
    "Kcnd2",  # Kv4.2, A-type K+ current, affects spike repolarization
    "Kcnd3",  # Kv4.3, A-type K+ current, affects spike repolarization
    "Kcnq2",  # Kv7.2, M-current, affects spike width and frequency
    "Kcnq3",  # Kv7.3, M-current, affects spike width and frequency
    "Kcnh1",  # Kv10.1/eag1, influences repolarization
    # Calcium-activated potassium channels
    "Kcnma1",  # KCa1.1/BK channel, large conductance, narrows AP
    "Kcnn1",  # KCa2.1/SK1, small conductance, affects AHP
    "Kcnn2",  # KCa2.2/SK2, small conductance, affects AHP
    # HCN channels (important for pacemaking and resonance)
    "Hcn1",  # Fast-activating HCN, affects rebound spiking
    "Hcn2",  # Slower-activating HCN, contributes to resonance
    "Hcn3",
    "Hcn4",  # Slowest-activating HCN, important in pacemaking
    # --- Leak channels and other important channels ---
    # K2P (Two-pore-domain potassium) leak channels
    "Kcnk1",  # TWIK-1, weak inward-rectifying K+ leak channel
    "Kcnk2",  # TREK-1, mechanosensitive and temperature-sensitive K2P channel
    "Kcnk3",  # TASK-1, pH-sensitive background K+ current
    "Kcnk4",  # TRAAK, mechanosensitive leak K+ channel
    "Kcnk9",  # TASK-3, regulates resting potential and excitability
    "Kcnk10",  # TREK-2, contributes to leak K+ conductance
    # Sodium leak channels and associated regulators
    "Nalcn",  # Na+ leak channel, controls resting Na+ conductance and excitability
    "Unc79",  # NALCN complex component
    "Unc80",  # NALCN complex component
    "Fam155a",  # Regulatory subunit of NALCN complex
    # Chloride leak channels
    "Clcn2",  # Voltage-gated Cl- channel, contributes to inhibitory tone
    # Inward-rectifier and GPCR-modulated K+ channels (affect resting potential)
    "Kcnj2",  # Kir2.1, stabilizes resting membrane potential
    "Kcnj10",  # Kir4.1, highly expressed in glia and neurons, contributes to K+ buffering
    "Kcnj3",  # GIRK1, G-protein-regulated inward-rectifying K+ channel
    "Kcnj6",  # GIRK2, major component of GIRK channels in CNS
    # --- Gap junctions and connexins ---
    # Gap junction–related genes (CNS)
    "Gja1",  # Connexin 43, major astrocytic gap junction protein
    "Gjb6",  # Connexin 30, co-expressed with Gja1 in glia
    "Gjd2",  # Connexin 36, neuronal gap junction, key for electrical synapses
    "Gjd3",  # Connexin 31.9, expressed in some CNS neurons
    "Panx1",  # Pannexin 1, forms large-pore channels in neurons and glia
    "Panx2",  # Pannexin 2, CNS-specific, neuron-enriched
    # --- Other common markers ---
    "Slc17a7",  # Excitatory marker
    "Gad1",  # Inhibitory marker
    "Sst",  # Somatostatin
    "Pvalb",  # Parvalbumin
    "Vip",  # Vasoactive intestinal peptide
    "Ndnf",  # NDNF marker
    "Lamp5",  # LAMP5 marker
    "Rorb",  # Layer 4 marker
    "Cux2",  # Layer 2/3 marker
    "Foxp2",  # Layer 6 marker
    "Ctgf",  # Layer 6b marker
    "Tshz2",  # Layer 5 ET marker
]


def extract_preselected_columns():
    """
    Read the log_normed_df.csv file from the data/LCNE-patchseq-ephys/seq directory,
    extract the preselected columns defined by SEQ_COLUMNS, and save it back to
    the data/seq directory.

    Returns:
        pd.DataFrame: The extracted dataframe with preselected columns
    """
    # Define the source and destination paths using absolute paths
    src_dir = Path(PACKAGE_DIRECTORY).resolve() / "../../data/LCNE-patchseq-ephys/seq"
    dst_dir = Path(PACKAGE_DIRECTORY).resolve() / "../../results/seq"

    # Ensure these paths are resolved to absolute paths
    src_dir = src_dir.resolve()
    dst_dir = dst_dir.resolve()

    src_file = src_dir / "log_normed_df_more_0620.csv"
    new_imp_file = src_dir / "length_normed_df_more_02122026.csv"  # Updated imputation data
    id_mapping_file = src_dir / "exp_component_ids_for_han_0620.csv"  # Aux file for mapping ids
    dst_file = dst_dir / "seq_preselected.csv"

    logger.info(f"Reading log-normalized data from {src_file}")

    # Check if source file exists
    if not src_file.exists():
        logger.error(f"Source file {src_file} does not exist")
        raise FileNotFoundError(f"Source file {src_file} does not exist")

    # Read the data - include the index column
    df = pd.read_csv(src_file, index_col=0)
    df_ids = pd.read_csv(id_mapping_file, index_col=0).rename(
        columns={"cell_id": "cell_specimen_id"}
    )

    # Reset the index to make it a regular column and rename it to 'exp_component_name'
    df = df.reset_index().rename(columns={"index": "exp_component_name"})

    # Merge in "cell_specimen_id"
    df = df.merge(df_ids, left_on="exp_component_name", right_on="exp_component_name.x", how="left")

    # Check which preselected columns exist in the dataframe
    available_columns = ["cell_specimen_id"] + [col for col in SEQ_COLUMNS if col in df.columns]
    missing_columns = [col for col in SEQ_COLUMNS if col not in df.columns]

    if missing_columns:
        logger.warning(f"The following columns are not available in the data: {missing_columns}")

    # Extract the available preselected columns
    df_extracted = df[available_columns].copy()

    # Scale imp_ML/DV/AP from MERFISH voxel units to µm (factor = 25)
    for col in ["imp_ML", "imp_DV", "imp_AP"]:
        if col in df_extracted.columns:
            df_extracted[col] = df_extracted[col] * 25

    # --- Merge new imputation columns from 02122026 ---
    if new_imp_file.exists():
        logger.info(f"Reading updated imputation data from {new_imp_file}")
        df_new = pd.read_csv(new_imp_file, index_col=0)
        df_new = df_new.reset_index().rename(columns={"index": "exp_component_name"})
        df_new = df_new.merge(
            df_ids, left_on="exp_component_name", right_on="exp_component_name.x", how="left"
        )
        # Rename columns to include the hyperparameter
        df_new = df_new.rename(
            columns={
                "imp_ML": "imp_ML_gaussian_sigma0p1_k100",
                "imp_DV": "imp_DV_gaussian_sigma0p1_k100",
                "imp_AP": "imp_AP_gaussian_sigma0p1_k100",
                "pc_gauss_sigma0p1": "imp_pseudoclusters_gaussian_sigma0p1_k100",
                "pc_gauss_sigma1": "imp_pseudoclusters_gaussian_sigma1_k100",
                "pc_softmax_tau0p1": "imp_pseudoclusters_softmax_tau0p1_k100",
                "pc_softmax_tau0p5": "imp_pseudoclusters_softmax_tau0p5_k100",
            }
        )
        new_cols = [
            "imp_ML_gaussian_sigma0p1_k100",
            "imp_DV_gaussian_sigma0p1_k100",
            "imp_AP_gaussian_sigma0p1_k100",
            "imp_pseudoclusters_gaussian_sigma0p1_k100",
            "imp_pseudoclusters_gaussian_sigma1_k100",
            "imp_pseudoclusters_softmax_tau0p1_k100",
            "imp_pseudoclusters_softmax_tau0p5_k100",
        ]
        available_new_cols = [c for c in new_cols if c in df_new.columns]
        missing_new_cols = [c for c in new_cols if c not in df_new.columns]
        if missing_new_cols:
            logger.warning(f"New imputation columns not found: {missing_new_cols}")
        df_new_sub = df_new[["cell_specimen_id"] + available_new_cols].copy()
        # Scale new imp_ML/DV/AP from MERFISH voxel units to µm (factor = 25)
        imp_cols = [
            "imp_ML_gaussian_sigma0p1_k100",
            "imp_DV_gaussian_sigma0p1_k100",
            "imp_AP_gaussian_sigma0p1_k100",
        ]
        for col in imp_cols:
            if col in df_new_sub.columns:
                df_new_sub[col] = df_new_sub[col] * 25
        df_extracted = df_extracted.merge(df_new_sub, on="cell_specimen_id", how="left")
        logger.info(f"Merged {len(available_new_cols)} new imputation columns from {new_imp_file.name}")  # noqa: E501
    else:
        logger.warning(f"New imputation file not found, skipping: {new_imp_file}")

    # Create the destination directory if it doesn't exist
    os.makedirs(dst_dir, exist_ok=True)

    # Save the extracted data
    logger.info(
        f"Saving extracted data (with {len(available_columns)} columns "
        f"and {len(df_extracted)} rows) to {dst_file}"
    )
    df_extracted.to_csv(dst_file, index=False)

    return df_extracted


if __name__ == "__main__":
    # Set up logging
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    # Run the extraction
    extract_preselected_columns()
