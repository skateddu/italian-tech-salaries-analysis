"""Pandas utility functions for descriptive statistics."""

import pandas as pd


def describe_complete(df: pd.DataFrame, column: str) -> pd.DataFrame:
    """Compute descriptive statistics including skewness and kurtosis.

    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    column : str
        Column name to describe.

    Returns
    -------
    pd.DataFrame
        Descriptive statistics with skew and kurtosis appended.
    """
    describe = df[column].describe().to_frame()
    skew_kurt = df[column].agg(func=["skew", "kurtosis"]).to_frame()
    return pd.concat(objs=[describe, skew_kurt]).transpose().round(decimals=2)


def describe_groups(df: pd.DataFrame, group_column: str, value_column: str) -> pd.DataFrame:
    """
    Compute descriptive statistics for groups in a DataFrame.
    
    Parameters
    ----------
    df : pd.DataFrame
        Input DataFrame.
    group_column : str
        Column name to group by.
    value_column : str
        Column name for which to compute statistics.

    Returns
    -------
    pd.DataFrame
        DataFrame with descriptive statistics for each group.
    """
    return df.groupby(by=group_column).agg(
        n=(value_column, "count"),
        mean=(value_column, "mean"),
        std=(value_column, "std"),
        min=(value_column, "min"),
        median=(value_column, "median"),
        max=(value_column, "max"),
    ).round(decimals=2)
