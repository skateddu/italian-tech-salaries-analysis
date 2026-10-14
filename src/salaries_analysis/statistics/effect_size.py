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
