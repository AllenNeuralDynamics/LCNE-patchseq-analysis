"""
Population analysis filters and confusion matrix computation.

This module contains query definitions and utilities for filtering and analyzing
cell populations based on various criteria including fluorescence status,
marker gene expression, and cell type classifications.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib_venn import venn3, venn3_circles

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

q_nucleus_present = '`jem-nucleus_post_patch` == "nucleus_present"'
q_nucleus_present_has_data = "`jem-nucleus_post_patch` == `jem-nucleus_post_patch`"

q_RNA_QC = "`gene_RNA_QC (log_normed)` == True"
q_RNA_QC_has_data = "`gene_RNA_QC (log_normed)` == `gene_RNA_QC (log_normed)`"

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
        "Marker Gene Dbh Positive": [if_marker_gene_dbh_positive, if_marker_gene_has_data],
        "MapMyCells Dbh Subclass": [if_mapmycells_dbh, if_mapmycells_has_data],
        "RNA QC": [df_meta.eval(q_RNA_QC), df_meta.eval(q_RNA_QC_has_data)],
        "Nucleus Present": [
            df_meta.eval(q_nucleus_present),
            df_meta.eval(q_nucleus_present_has_data),
        ],
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


def plot_venn_three_filters(condition_mapper, to_compare: list, ax=None):
    """
    Plot a Venn diagram for three boolean filters from condition_mapper.

    Parameters:
    -----------
    condition_mapper : dict
        Dictionary mapping filter names to [positive_condition, has_data_condition] pairs
    to_compare : list
        List of 3 filter names to compare (must be keys in condition_mapper)
    ax : matplotlib.axes.Axes, optional
        Axes to plot on. If None, creates new figure and axes.

    Returns:
    --------
    matplotlib.axes.Axes
        The axes containing the plot
    """
    if len(to_compare) != 3:
        raise ValueError("to_compare must contain exactly 3 filter names")

    # Get the positive conditions (first element) for each filter
    filter1 = condition_mapper[to_compare[0]][0]  # positive condition
    filter2 = condition_mapper[to_compare[1]][0]  # positive condition
    filter3 = condition_mapper[to_compare[2]][0]  # positive condition

    # Convert boolean arrays to sets of indices
    def to_set(f):
        if isinstance(f, (pd.Series, np.ndarray)) and f.dtype == bool:
            return set(np.flatnonzero(f))
        elif isinstance(f, (pd.Series, np.ndarray)):
            return set(f)
        elif isinstance(f, set):
            return f
        else:
            return set(list(f))

    set1, set2, set3 = to_set(filter1), to_set(filter2), to_set(filter3)

    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6), dpi=300)

    # Use the filter names as labels
    labels = to_compare
    v = venn3([set1, set2, set3], set_labels=labels, ax=ax)
    c = venn3_circles([set1, set2, set3], ax=ax)

    # Set edge color and style
    for i, color in enumerate(("black", "blue", "green")):
        c[i].set_edgecolor(color)
        v.get_label_by_id(["A", "B", "C"][i]).set_color(color)

    # Clear all patch color
    for patch in v.patches:
        if patch:  # Some patches might be None
            patch.set_facecolor("none")

    return ax


def plot_venn_summary(
    df_lists,
    compare_lists=[
        ["Fluorescence", "Marker Gene Dbh Positive", "MapMyCells Dbh Subclass"],
        ["Fluorescence", "Marker Gene All Positive", "MapMyCells Dbh Subclass"],
        ["Fluorescence", "Marker Gene Any Positive", "MapMyCells Dbh Subclass"],
        ["RNA QC", "Nucleus Present", "MapMyCells Dbh Subclass"],
    ],
):
    """
    Plot a summary of Venn diagrams for multiple DataFrames and filter conditions.

    Parameters:
    -----------
    df_lists : list of pd.DataFrame
        List of DataFrames containing metadata for different cell populations
    compare_lists : list of list of str, optional
        List of filter names to compare in Venn diagrams. Default includes common filters.

    Returns:
    --------
    matplotlib.figure.Figure
        Figure containing the Venn diagrams for each comparison.
    """
    fig, ax = plt.subplots(len(compare_lists), 2, figsize=(12, len(compare_lists) * 4), dpi=200)

    for i, df in enumerate(df_lists):
        condition_mapper = create_filter_conditions(df)
        for j, filters in enumerate(compare_lists):
            plot_venn_three_filters(condition_mapper, filters, ax=ax[j, i])

    plt.tight_layout()
    return fig
