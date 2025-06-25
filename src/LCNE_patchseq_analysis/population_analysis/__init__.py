"""
Population analysis module for LCNE patchseq analysis.

This module provides utilities for filtering and analyzing cell populations
based on various criteria including fluorescence status, marker gene expression,
and cell type classifications.
"""

from .filters import (
    q_fluorescence,
    q_fluorescence_has_data,
    q_marker_gene_any_positive,
    q_marker_gene_all_positive,
    q_marker_gene_dbh_positive,
    q_marker_gene_has_data,
    q_mapmycells_dbh,
    q_mapmycells_has_data,
    create_filter_conditions,
    compute_confusion_matrix,
)

__all__ = [
    'q_fluorescence',
    'q_fluorescence_has_data',
    'q_marker_gene_any_positive',
    'q_marker_gene_all_positive',
    'q_marker_gene_dbh_positive',
    'q_marker_gene_has_data',
    'q_mapmycells_dbh',
    'q_mapmycells_has_data',
    'create_filter_conditions',
    'compute_confusion_matrix',
]
