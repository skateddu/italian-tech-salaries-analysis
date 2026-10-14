"""Effect size measures for statistical comparisons."""

import numpy as np
import pandas as pd


def cohens_d(group1: pd.Series, group2: pd.Series) -> float:
    """Compute Cohen's d effect size between two groups.

    Parameters
    ----------
    group1 : pd.Series
        First group of values.
    group2 : pd.Series
        Second group of values.

    Returns
    -------
    float
        Cohen's d effect size.
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = group1.var(), group2.var()
    pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (group1.mean() - group2.mean()) / pooled_std


def partial_eta_squared(anova_table: pd.DataFrame) -> pd.Series:
    """Compute partial eta squared for each term of an ANOVA table.

    Partial eta squared is the share of variance explained by a term after
    removing the variance explained by the other terms:
    ``SS_effect / (SS_effect + SS_residual)``. Unlike regression coefficients,
    it is on the same scale for continuous and categorical factors, so it can
    be used to rank factors by importance.

    Parameters
    ----------
    anova_table : pd.DataFrame
        ANOVA table as returned by ``statsmodels.stats.anova.anova_lm``, with
        a ``sum_sq`` column and a ``Residual`` row.

    Returns
    -------
    pd.Series
        Partial eta squared for each term, sorted in descending order.
    """
    ss_residual = anova_table.loc["Residual", "sum_sq"]
    effects = anova_table.drop(index="Residual")["sum_sq"]
    return (
        (effects / (effects + ss_residual)).sort_values(ascending=False).rename("partial_eta_sq")
    )
