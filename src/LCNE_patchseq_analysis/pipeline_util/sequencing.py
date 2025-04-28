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
    # Coordinates and clusters
    "imp_ML",
    "imp_DV",
    "imp_AP",
    "imp_pseudoclusters",
    # Noradrenergic markers
    "Dbh",  # Dopamine beta-hydroxylase
    "Th",  # Tyrosine hydroxylase
    "Slc18a2",  # Vesicular monoamine transporter 2 (VMAT2)
    "Slc6a2",  # Norepinephrine transporter (NET)
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
    "Hcn4",  # Slowest-activating HCN, important in pacemaking
    # Other common markers
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

    src_file = src_dir / "log_normed_df.csv"
    id_mapping_file = src_dir / "exp_component_ids_for_han.csv"  # Aux file for mapping ids
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
