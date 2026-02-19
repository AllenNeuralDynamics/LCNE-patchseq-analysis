# LCNE-patchseq-analysis

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-86.4%25-yellow)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.9-blue?logo=python)

Library for LCNE-patchseq data analysis

## Overview
Here is the overall workflow of patchseq analysis for the LC-NE project

<img width="1540" alt="image" src="https://github.com/user-attachments/assets/596f8c82-8bc7-45c5-b4c1-facc03265a7d" />

This repository maintains all the green parts in the above diagram:
1. [`src/LCNE_patchseq_analysis/pipeline_util`](https://github.com/AllenNeuralDynamics/LCNE-patchseq-analysis/tree/main/src/LCNE_patchseq_analysis/pipeline_util): upload data and metadata from various locations to cloud
2. [`.../efel`](https://github.com/AllenNeuralDynamics/LCNE-patchseq-analysis/tree/main/src/LCNE_patchseq_analysis/efel): extract ephys features using the [eFEL library](https://efel.readthedocs.io/en/latest/eFeatures.html).

## Detailed workflow
<img width="1240" alt="image" src="https://github.com/user-attachments/assets/f771ced3-5ec5-4607-a2cb-be2b4993dd23" />


## Reproducing the eFEL pipeline end-to-end in CodeOcean

The eFEL pipeline can be run reproducibly inside a CodeOcean capsule. When the
environment variable `CO_COMPUTATION_ID` is present (set automatically by
CodeOcean at runtime), the package switches all paths to the CodeOcean layout:

| Path | Purpose |
|------|---------|
| `/data/LCNE-patchseq-ephys/raw/` | NWB files, JSON outputs, and Brian's metadata spreadsheet |
| `/data/LCNE-patchseq-ephys/morphology/` | Morphology files |
| `/results/` | All pipeline outputs (features, plots, cell_stats) |

### Prerequisites

1. **Attach the dataset** — the capsule must have the `LCNE-patchseq-ephys`
   dataset (id `68ef27d7-9d95-40ce-9e40-7de93dccf5f8`) attached and mounted at
   `LCNE-patchseq-ephys` (already configured in `.codeocean/datasets.json`).

2. **Dataset contents** — the dataset must include:
   - `raw/Ephys_Roi_Result_<id>/` — per-cell NWB and JSON files
   - `raw/df_metadata_merged.csv` — Brian's master spreadsheet
   - `raw/df_metadata_merged_20250409.csv` — xyz coordinate supplement

3. **Install eFEL extras** — the `efel` and `tables` packages are optional
   extras not included in the base install. Either update the `Dockerfile` pip
   install line to use the `[efel]` extra:
   ```dockerfile
   RUN pip install -U --no-cache-dir \
       "git+https://github.com/AllenNeuralDynamics/LCNE-patchseq-analysis.git@<commit>#egg=LCNE-patchseq-analysis[efel]"
   ```
   or add them explicitly:
   ```dockerfile
   RUN pip install -U --no-cache-dir efel tables tqdm
   ```

### Running the pipeline

Set `code/run` to invoke the pipeline entry point:

```bash
#!/usr/bin/env bash
set -ex
python -m LCNE_patchseq_analysis.efel.pipeline
```

Click **Reproducible Run**. The pipeline will:

1. **Extract eFEL features** — reads each cell's NWB file from
   `/data/LCNE-patchseq-ephys/raw/`, writes per-cell `.h5` files to
   `/results/features/`.
2. **Generate sweep plots** — writes per-sweep PNG files to
   `/results/plots/<roi_id>/`.
3. **Compute cell-level statistics** — aggregates features across sweeps,
   merges with Brian's spreadsheet, and writes:
   - `/results/cell_stats/cell_level_stats.csv` — main summary table
   - `/results/cell_stats/cell_level_*.pkl` — representative spike waveforms
   - `/results/cell_stats/<roi_id>_cell_summary.png` — per-cell summary plots

### Verifying the mode at runtime

The logger will print a banner at startup confirming CodeOcean mode:

```
============================================================
Running in CodeOcean mode
  RAW data  : /data/LCNE-patchseq-ephys/raw
  Results   : /results
  Metadata  : loaded from results folder (not S3)
============================================================
```

And again when the pipeline `__main__` block starts:

```
============================================================
Pipeline running in CODEOCEAN mode
  Input : /data/LCNE-patchseq-ephys/raw
  Output: /results
============================================================
```

### Local / S3 mode

Without `CO_COMPUTATION_ID` set, all paths fall back to the developer-local
defaults defined in `src/LCNE_patchseq_analysis/__init__.py` and
`cell_level_stats.csv` is fetched from the public S3 bucket instead.


## Loading the master metadata table (df_meta)

All figure scripts share a single entry point for loading data:

```python
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata

df_meta = load_ephys_metadata(
    if_from_s3=True,           # True: load eFEL stats; False: Brian's spreadsheet only
    if_with_seq=True,          # merge gene expression + MapMyCells cell-type calls
    if_with_morphology=True,   # merge morphology features
    combine_roi_ids=True,      # unify LIMS and spreadsheet ROI IDs
)
```

**What `if_from_s3=True` returns** — the full merged master table ("df_meta") with:

| Column group | Source | Prefix |
|---|---|---|
| Ephys passive properties, QC | eFEL pipeline output | `ipfx_*` |
| Spike-waveform eFEL features | eFEL pipeline output | `efel_*` |
| Injection region, LC targeting, coordinates | Brian's spreadsheet | _(no prefix)_ |
| Gene expression (log-normalised) | `seq_preselected.csv` on S3 | `gene_*` |
| Cell-type classification | MapMyCells results on S3 | `mapmycells_*` |
| Morphology metrics | `LC_patchseq_RawFeatureWide.csv` on S3 | `morphology_*` |

In **CodeOcean mode** (`CO_COMPUTATION_ID` set), calling `load_ephys_metadata(if_from_s3=True)` first
checks for `/results/cell_stats/cell_level_stats.csv` produced by the eFEL pipeline; if that file
exists it is used directly, otherwise it falls back to S3. This means the figures script can run
either after the eFEL pipeline (fully self-contained) or standalone against the public S3 data.

**What `if_from_s3=False` returns** — Brian's master spreadsheet only
(`raw/df_metadata_merged.csv`), filtered to cells present in the spreadsheet
or LIMS. This is used internally by the eFEL pipeline itself to enumerate
which cells to process.

**Recommended filter** for main figures (excludes reporter-negative and
thalamus-injected cells):

```python
from LCNE_patchseq_analysis.figures import GLOBAL_FILTER, GENE_FILTER

df_filtered = df_meta.query(GLOBAL_FILTER)          # all projection targets
df_dbh     = df_meta.query(GENE_FILTER)             # DBH+ cells only (for seq figures)
```


## Generating main figures

All figure scripts live in
[`src/LCNE_patchseq_analysis/figures/`](https://github.com/AllenNeuralDynamics/LCNE-patchseq-analysis/tree/main/src/LCNE_patchseq_analysis/figures)
and can be run directly or imported as functions.

### Individual figure scripts

| Script | Entry point | Key output |
|--------|-------------|-----------|
| `figures/main_imputation.py` | `main_imputation(df_meta, GENE_FILTER)` | Imputed pseudocluster comparison |
| `figures/main_pca_tau.py` | `figure_spike_pca(df_meta, df_spikes)` | Spike-waveform PCA + tau panels |

All functions accept an optional `ax` argument for embedding panels in a larger
figure, and `if_save_figure=True` (default) to write PNG + SVG to `RESULTS_DIRECTORY`.

### Spike waveforms (needed for PCA figure)

Average spike waveforms per cell are loaded separately from S3:

```python
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes

df_spikes = get_public_representative_spikes("average")  # also: "first", "second", "last"
```


## The Panel app
The Panel app has been migrated to [LCNE-patchseq-viz](https://github.com/AllenNeuralDynamics/LCNE-patchseq-viz.git).

- **Live app**: [https://hanhou-patchseq.hf.space/patchseq_panel_viz](https://hanhou-patchseq.hf.space/patchseq_panel_viz)
- **Source**: [https://github.com/AllenNeuralDynamics/LCNE-patchseq-viz](https://github.com/AllenNeuralDynamics/LCNE-patchseq-viz)


