"""
Population analysis module for LCNE patchseq analysis.

This module provides utilities for filtering and analyzing cell populations
based on various criteria including fluorescence status, marker gene expression,
and cell type classifications, as well as spike analysis utilities.
"""

from .filters import (
    compute_confusion_matrix,
    create_filter_conditions,
)
from .spikes import (
    extract_representative_spikes,
    extract_simple_representative_spikes,
    normalize_data,
    normalize_spike_waveform,
)

__all__ = [
    "create_filter_conditions",
    "compute_confusion_matrix",
    "normalize_data",
    "normalize_spike_waveform",
    "extract_representative_spikes",
    "extract_simple_representative_spikes",
]
