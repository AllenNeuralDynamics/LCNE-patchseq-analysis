# LCNE-patchseq-analysis

[![License](https://img.shields.io/badge/license-MIT-brightgreen)](LICENSE)
![Code Style](https://img.shields.io/badge/code%20style-black-black)
[![semantic-release: angular](https://img.shields.io/badge/semantic--release-angular-e10079?logo=semantic-release)](https://github.com/semantic-release/semantic-release)
![Interrogate](https://img.shields.io/badge/interrogate-86.0%25-yellow)
![Coverage](https://img.shields.io/badge/coverage-100%25-brightgreen?logo=codecov)
![Python](https://img.shields.io/badge/python->=3.9-blue?logo=python)

This repository is the **main entry point for reproducing the LCNE-patchseq analysis** associated with our paper. It contains the full pipeline for extracting electrophysiological features from patch-seq recordings in locus coeruleus norepinephrine (LC-NE) neurons, merging transcriptomic and morphological data, and generating the paper's main figures. A fully reproducible run — with identical data, software environment, and code — is available as a [Code Ocean capsule](https://codeocean.allenneuraldynamics.org/capsule/1699143/tree). An interactive [Panel app](https://hanhou-patchseq.hf.space/patchseq_panel_viz) is provided for exploring the dataset.

## Resources

| Resource | Description | Link |
|---|---|---|
| **Analysis code** (this repo) | Source code for the eFEL pipeline, figure scripts, and data utilities | [AllenNeuralDynamics/LCNE-patchseq-analysis](https://github.com/AllenNeuralDynamics/LCNE-patchseq-analysis) |
| **Code Ocean capsule** | Fully reproducible computational capsule — data, environment, and code used for the paper | [Capsule #1699143](https://codeocean.allenneuraldynamics.org/capsule/1699143/tree) |
| **Interactive visualization app** | Panel app for exploring single-cell ephys, transcriptomics, and morphology data | [hanhou-patchseq.hf.space/patchseq_panel_viz](https://hanhou-patchseq.hf.space/patchseq_panel_viz) |
| **Visualization source code** | Source code for the Panel app, deployed on Hugging Face Spaces | [AllenNeuralDynamics/LCNE-patchseq-viz](https://github.com/AllenNeuralDynamics/LCNE-patchseq-viz) |
| **ipfx** (upstream) | Allen Institute library for ephys feature extraction and QC, used by the snakemake-ipfx pipeline | [AllenInstitute/ipfx](https://github.com/AllenInstitute/ipfx) |
| **snakemake-ipfx** (upstream) | Snakemake pipeline for running ipfx at scale; produces the NWB files ingested here ([Gouwens et al. 2021](https://elifesciences.org/articles/65482)) | [AllenInstitute/snakemake_ephys](https://github.com/AllenInstitute/snakemake_ephys) |

## Overview

<img width="1462" height="520" alt="image" src="https://github.com/user-attachments/assets/da1474d7-fa3c-447d-ab7e-7090858b65fd" />

The diagram above shows the full patchseq analysis workflow for the LC-NE project. **This repository covers only the green arrows.** The grey upstream steps — LIMS data management, the snakemake-ipfx ephys QC pipeline ([AllenInstitute/ipfx](https://github.com/AllenInstitute/ipfx), [AllenInstitute/snakemake_ephys](https://github.com/AllenInstitute/snakemake_ephys), [Gouwens et al. 2021](https://elifesciences.org/articles/65482)), and sequencing data processing — are outside the scope of this repository; this pipeline takes their outputs as inputs.

- [`pipeline_util`](https://github.com/AllenNeuralDynamics/LCNE-patchseq-analysis/tree/main/src/LCNE_patchseq_analysis/pipeline_util) — ingests raw data and metadata from various sources and uploads them to cloud storage (S3; **replaced by the mounted dataset at `/data/` in Code Ocean mode**)
- [`efel`](https://github.com/AllenNeuralDynamics/LCNE-patchseq-analysis/tree/main/src/LCNE_patchseq_analysis/efel) — extracts electrophysiological features from NWB files using the [eFEL library](https://efel.readthedocs.io/en/latest/eFeatures.html), then aggregates them into a cell-level summary table (written to S3 locally; **written to `/results/` in Code Ocean mode**)
- [`figures`](https://github.com/AllenNeuralDynamics/LCNE-patchseq-analysis/tree/main/src/LCNE_patchseq_analysis/figures) — generates the paper's main figures by merging ephys features with transcriptomic and morphological data (reads summary table from S3; **reads from `/results/` in Code Ocean mode**)

## Detailed workflow

<img width="1240" alt="image" src="https://github.com/user-attachments/assets/f771ced3-5ec5-4607-a2cb-be2b4993dd23" />

The diagram above shows the step-by-step data flow through the pipeline: from raw NWB files and metadata spreadsheets, through eFEL feature extraction and multi-modal merging (transcriptomics via MapMyCells, morphology), to the final outputs consumed by the figure scripts and the Panel app. Any interaction with S3 shown in the diagram is **replaced by the local Code Ocean filesystem** (`/data/` for inputs, `/results/` for outputs) when running inside the Code Ocean capsule.


## Reproducing our work in Code Ocean

> **Tip:** Before diving in, we highly encourage you to explore the dataset interactively via the [Panel app](https://hanhou-patchseq.hf.space/patchseq_panel_viz) first.

All analyses can be reproduced from the [Code Ocean capsule](https://codeocean.allenneuraldynamics.org/capsule/1699143/tree). The capsule bundles the data, environment, and code — no setup required.

1. **Generate the main figures** (default) — click **Reproducible Run**. The capsule will load the pre-computed eFEL feature table from the attached dataset and run all figure scripts directly.

2. **Re-run the full pipeline** — to redo all green-arrow steps (eFEL feature extraction → cell-level statistics → figures) from scratch within the capsule, trigger a Reproducible Run with the app argument `rerun_efel_pipeline=1`. ⚠️ *This will take several hours.*

3. **Interactive debugging** — open the capsule in **VS Code** (via the Code Ocean IDE), then install the package in editable mode:
   ```bash
   pip install -e .
   ```
   You can then edit and re-run any part of the pipeline interactively.

Alternatively, if you prefer to work outside of Code Ocean, see the standalone instructions below.

## Reproducing standalone

The figures can also be reproduced locally or in any notebook environment — all data are fetched directly from the public S3 bucket, so no local data download is required. Install the package via PyPI (`pip install LCNE-patchseq-analysis`) and run:


```python
#!pip install LCNE-patchseq-analysis
from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.pipeline_util.s3 import get_public_representative_spikes

from LCNE_patchseq_analysis.figures import GLOBAL_FILTER, GENE_FILTER
from LCNE_patchseq_analysis.figures.main_pca_tau import figure_spike_pca
from LCNE_patchseq_analysis.figures.main_imputation import main_imputation

# -- Physiological properties --
# Load merged metadata (eFEL features + key metadata + transcriptomics) from public S3
df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
df_meta = df_meta.query(GLOBAL_FILTER)  # exclude reporter-negative and thalamus-injected cells

# Load per-cell average spike waveforms and generate spike PCA figure
df_spikes = get_public_representative_spikes("average")
fig, axes_dict, results = figure_spike_pca(df_meta, df_spikes, filtered_df_meta=df_meta)

# -- Gene imputations --
main_imputation(df_meta, GENE_FILTER)
```


## Two modes of running the pipeline

The two reproduction paths above correspond to two runtime modes of the package, which differ in where data is read from and where results are written:

| | **Code Ocean mode** | **Local / developer mode** |
|---|---|---|
| **Trigger** | `CO_COMPUTATION_ID` env var is set automatically by Code Ocean at runtime | Running locally without `CO_COMPUTATION_ID` |
| **Input data** | Dataset `68ef27d7-9d95-40ce-9e40-7de93dccf5f8` mounted at `/data/LCNE-patchseq-ephys/` | `s3://aind-scratch-data/aind-patchseq-data/` |
| **Results** | Written to `/results/` within the capsule | Written locally; key outputs (e.g. `cell_level_stats.csv`, spike waveforms) pushed to the public S3 bucket |
| **Panel app** | Not connected — capsule results are self-contained | S3 outputs are what the [Panel app](https://hanhou-patchseq.hf.space/patchseq_panel_viz) reads from |

In short: **Code Ocean mode** is for reproducibility (everything stays inside the capsule), while **local mode** is for active development and feeds results into the live visualization app via S3.


## The Panel app
The Panel app has been migrated to [LCNE-patchseq-viz](https://github.com/AllenNeuralDynamics/LCNE-patchseq-viz.git).

- **Live app**: [https://hanhou-patchseq.hf.space/patchseq_panel_viz](https://hanhou-patchseq.hf.space/patchseq_panel_viz)
- **Source**: [https://github.com/AllenNeuralDynamics/LCNE-patchseq-viz](https://github.com/AllenNeuralDynamics/LCNE-patchseq-viz)


