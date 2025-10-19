"""anova
=================
Utilities to perform per-feature ANCOVA / ANOVA across many electrophysiology
features. Model form used:

    feature ~ C(injection_region) + y

Where ``injection_region`` is treated as categorical and ``y`` is a continuous
coordinate (dorsal-ventral). For each feature a Type II ANOVA table is
generated and key statistics (F, p, sums of squares, partial eta squared) are
collected into a tidy DataFrame.

"""

import logging
logger = logging.getLogger(__name__)

from typing import Iterable, List

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multitest import multipletests

from LCNE_patchseq_analysis.data_util.metadata import load_ephys_metadata
from LCNE_patchseq_analysis.figures import DEFAULT_EPHYS_FEATURES


def anova_features(
    df: pd.DataFrame,
    features: Iterable[str],
    cat_col: str = "injection region",
    cont_col: str = "y",
    adjust_p: bool = False,
    anova_typ: int = 2,
) -> pd.DataFrame:
    """Run per-feature ANCOVA (additive model) and return tidy statistics.

    Each feature is modeled as: ``feature ~ C(cat_col) + cont_col``.

    Parameters
    ----------
    df : DataFrame
        Input data containing features and predictor columns.
    features : Iterable[str]
        Names of columns in ``df`` to analyze (dependent variables).
    cat_col : str, default 'injection region'
        Categorical predictor column name.
    cont_col : str, default 'y'
        Continuous predictor column name.
    adjust_p : bool, default False
        If True apply Benjamini-Hochberg FDR correction across all (feature, term) p-values.
    anova_typ : int, default 2
        Type passed to ``statsmodels.stats.anova_lm`` (Type II recommended for unbalanced designs).

    Returns
    -------
    DataFrame
        Tidy table with columns:
        ['feature','term','df','sum_sq','mean_sq','F','p','partial_eta_sq',('p_adj','significant')?]
    """
    records: List[dict] = []

    for feat in features:
        if feat not in df.columns:
            logger.warning("Feature '%s' not in DataFrame; skipping", feat)
            continue
        needed = [feat, cat_col, cont_col]
        sub = df[needed].dropna()
        if sub.empty:
            logger.warning("Feature '%s' has no non-null rows after dropping NA; skipping", feat)
            continue

        formula = f'Q("{feat}") ~ C(Q("{cat_col}")) + Q("{cont_col}")'
        try:
            model = ols(formula, data=sub).fit()
            aov = sm.stats.anova_lm(model, typ=anova_typ)
        except Exception as exc:  # pragma: no cover - defensive
            logger.error("ANOVA failed for feature '%s': %s", feat, exc)
            continue

        residual_ss = aov.loc["Residual", "sum_sq"] if "Residual" in aov.index else np.nan
        for term in aov.index:
            if term == "Residual":
                continue
            row = aov.loc[term]
            sum_sq = row["sum_sq"]
            df_term = row["df"]
            mean_sq = sum_sq / df_term if df_term else np.nan
            f_val = row.get("F", np.nan)
            p_val = row.get("PR(>F)", np.nan)
            partial_eta_sq = (
                sum_sq / (sum_sq + residual_ss) if np.isfinite(residual_ss) and (sum_sq + residual_ss) > 0 else np.nan
            )
            records.append(
                {
                    "feature": feat,
                    "term": term,
                    "df": df_term,
                    "sum_sq": sum_sq,
                    "mean_sq": mean_sq,
                    "F": f_val,
                    "p": p_val,
                    "partial_eta_sq": partial_eta_sq,
                }
            )

    out = pd.DataFrame.from_records(records)
    if out.empty:
        return out

    # Drop any rows where p is NaN or not finite (user requested simpler behavior)
    out = out[out["p"].notna() & np.isfinite(out["p"])].copy()

    if adjust_p and not out.empty:
        # Adjust within each term separately
        out.reset_index(drop=True, inplace=True)
        out["p_adj"] = np.nan
        out["significant"] = False
        unique_terms = list(out["term"].unique())
        for term in unique_terms:
            term_idx = out.index[out["term"] == term]
            if len(term_idx) == 0:
                continue
            # Build numeric p-value array explicitly to satisfy type checkers
            pvals = np.array([float(x) for x in out.loc[term_idx, "p"]], dtype=float)
            if pvals.size == 0:
                continue
            reject, p_adj, _, _ = multipletests(pvals, method="fdr_bh")
            out.loc[term_idx, "p_adj"] = p_adj
            out.loc[term_idx, "significant"] = reject

    # Sort according to the lowest p-value between the two terms for each feature
    features_sorted = (
        out.groupby("feature")["p"].min().sort_values().index.tolist()
    )
    out = out.set_index("feature").loc[features_sorted].reset_index()
    return out


if __name__ == "__main__":
    # --- Fig 3a. Sagittal view of LC-NE cells colored by projection ---
    logger.info("Loading metadata...")
    df_meta = load_ephys_metadata(if_from_s3=True, if_with_seq=True)
    logger.info(f"Loaded metadata with shape: {df_meta.shape}")

    from LCNE_patchseq_analysis.figures import GLOBAL_FILTER, GENE_FILTER

    # Run ANOVA on selected ephys features
    result_ipfx = anova_selected_ephys_features(df_meta, filter_query=GLOBAL_FILTER)
    logger.info("ANOVA completed on %d features", result_ipfx["feature"].nunique())
    print(result_ipfx.head())

    # Run ANOVA on gene expression
    df_filtered = df_meta.query(GENE_FILTER)
    result_gene = anova_gene(
        df_filtered,
        filter_query=GENE_FILTER
    )