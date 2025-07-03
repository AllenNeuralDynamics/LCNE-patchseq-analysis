"""
Population analysis filters and confusion matrix computation.

This module contains query definitions and utilities for filtering and analyzing
cell populations based on various criteria including fluorescence status,
marker gene expression, and cell type classifications.
"""

import pandas as pd
from matplotlib_venn import venn3, venn3_circles
import matplotlib.pyplot as plt
import numpy as np

# Query definitions for different cell filtering criteria
q_fluorescence = '`jem-status_reporter` == "Positive"'
q_fluorescence_has_data = (
    "`jem-status_reporter` == `jem-status_reporter`"  # True for non-NaN values
)

q_marker_gene_any_positive = (
    "`gene_Dbh (log_normed)` > 0 or `gene_Th (log_normed)` > 0 or "
    "`gene_Slc18a2 (log_normed)` > 0 or `gene_Slc6a2 (log_normed)` > 0"
)
q_marker_gene_all_positive = (
    "`gene_Dbh (log_normed)` > 0 and `gene_Th (log_normed)` > 0 and "
    "`gene_Slc18a2 (log_normed)` > 0 and `gene_Slc6a2 (log_normed)` > 0"
)
q_marker_gene_dbh_positive = "`gene_Dbh (log_normed)` > 0"
q_marker_gene_has_data = (
    "`gene_Dbh (log_normed)` == `gene_Dbh (log_normed)`"  # True for non-NaN values
)
q_mapmycells_dbh = 'mapmycells_subclass_name.str.contains("DBH", case=False, na=False)'
q_mapmycells_has_data = (
    "mapmycells_subclass_name == mapmycells_subclass_name"  # True for non-NaN values
)
q_retro = '`injection region` != "Non-Retro"'


def create_filter_conditions(df_meta):
    """
    Create boolean filter conditions based on the metadata DataFrame.

    Parameters:
    -----------
    df_meta : pd.DataFrame
        Metadata DataFrame containing cell information

    Returns:
    --------
    dict
        Dictionary mapping filter names to [positive_condition, has_data_condition] pairs
    """
    if_fluorescence_positive = df_meta.eval(q_fluorescence)
    if_fluorescence_has_data = df_meta.eval(q_fluorescence_has_data)
    if_marker_gene_any_positive = df_meta.eval(q_marker_gene_any_positive)
    if_marker_gene_all_positive = df_meta.eval(q_marker_gene_all_positive)
    if_marker_gene_dbh_positive = df_meta.eval(q_marker_gene_dbh_positive)
    if_marker_gene_has_data = df_meta.eval(q_marker_gene_has_data)
    if_mapmycells_dbh = df_meta.eval(q_mapmycells_dbh)
    if_mapmycells_has_data = df_meta.eval(q_mapmycells_has_data)

    condition_mapper = {  # [condition for positive, condition for having data]
        "Fluorescence": [if_fluorescence_positive, if_fluorescence_has_data],
        "Marker Gene Any Positive": [if_marker_gene_any_positive, if_marker_gene_has_data],
        "Marker Gene All Positive": [if_marker_gene_all_positive, if_marker_gene_has_data],
        "Marker Gene DBH Positive": [if_marker_gene_dbh_positive, if_marker_gene_has_data],
        "Is DBH Subclass": [if_mapmycells_dbh, if_mapmycells_has_data],
    }

    return condition_mapper


def compute_confusion_matrix(condition_mapper, name1, name2):
    """
    Compute confusion matrix between two filter conditions.

    Parameters:
    -----------
    condition_mapper : dict
        Dictionary mapping filter names to [positive_condition, has_data_condition] pairs
    name1 : str
        Name of the first filter condition
    name2 : str
        Name of the second filter condition

    Returns:
    --------
    pd.DataFrame
        Confusion matrix as a DataFrame
    """
    if_1 = condition_mapper[name1][0]
    if_2 = condition_mapper[name2][0]
    both_has_data = condition_mapper[name1][1] & condition_mapper[name2][1]
    pos_pos = if_1 & if_2 & both_has_data
    pos_neg = if_1 & ~if_2 & both_has_data
    neg_pos = ~if_1 & if_2 & both_has_data
    neg_neg = ~if_1 & ~if_2 & both_has_data
    unknown = ~both_has_data
    confusion_matrix = pd.DataFrame(
        {
            f"`{name1}` (+)": [pos_pos.sum(), pos_neg.sum()],
            f"`{name1}` (-)": [neg_pos.sum(), neg_neg.sum()],
        },
        index=[f"`{name2}` (+)", f"`{name2}` (-)"],
    ).T

    print(f"`{name1}` does not have data: {(~condition_mapper[name1][1]).sum()}")
    print(f"`{name2}` does not have data: {(~condition_mapper[name2][1]).sum()}")
    print(f"Any of them does not have data: {unknown.sum()}")
    return confusion_matrix

def plot_venn_three_filters(filter1, filter2, filter3, labels=("Filter 1", "Filter 2", "Filter 3"), ax=None):
    """
    Plot a Venn diagram for three boolean filters or sets.
    Accepts either boolean arrays (same length) or sets of indices.
    """
    # Convert boolean arrays to sets of indices if needed
    def to_set(f):
        if isinstance(f, (pd.Series, np.ndarray)) and f.dtype == bool:
            return set(np.where(f)[0])
        elif isinstance(f, (pd.Series, np.ndarray)):
            return set(f)
        elif isinstance(f, set):
            return f
        else:
            return set(list(f))
    set1, set2, set3 = to_set(filter1), to_set(filter2), to_set(filter3)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    v = venn3([set1, set2, set3], set_labels=labels, ax=ax)
    c = venn3_circles([set1, set2, set3], ax=ax)

    # Set edge color and style
    for i, color in enumerate(("black", "blue", "green")):
        c[i].set_edgecolor(color)
        v.get_label_by_id(["A", "B", "C"][i]).set_color(color)
    
    # Clear all patch color
    for patch in v.patches:
        if patch:  # Some patches might be None
            patch.set_facecolor('none')


    return ax
