"""Helper functions for common plot patterns."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

STYLE_PATH = Path(__file__).resolve().parent.parent / "config.mplstyle"


def plot_salary_boxplot(
    data: pd.DataFrame,
    x: str,
    y: str = "salary",
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Salary (€)",
    order: list[str] | None = None,
    ylim: tuple[int, int] | None = (0, 120_000),
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """Draw a boxplot of salary by a categorical variable.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataframe.
    x : str
        Column name for the categorical axis.
    y : str
        Column name for the continuous axis.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    order : list[str] or None
        Category display order.
    ylim : tuple[int, int] or None
        Y-axis limits.
    figsize : tuple[int, int]
        Figure size (width, height).
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.boxplot(data=data, x=x, y=y, order=order, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if ylim is not None:
        ax.set_ylim(*ylim)
    plt.tight_layout()
    plt.show()


def plot_heatmap(
    data: pd.DataFrame,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    fmt: str = ",.0f",
    cmap: str = "YlOrRd",
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """Draw an annotated heatmap.

    Parameters
    ----------
    data : pd.DataFrame
        Pivot table or matrix to display.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    fmt : str
        Number format for annotations.
    cmap : str
        Colormap name.
    figsize : tuple[int, int]
        Figure size (width, height).
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(data, annot=True, fmt=fmt, linewidths=1, cmap=cmap, ax=ax)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()


def plot_horizontal_barplot(
    data: pd.DataFrame,
    x: str,
    y: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: tuple[int, int] = (10, 5),
) -> None:
    """Draw a horizontal bar plot with standard deviation error bars.

    Parameters
    ----------
    data : pd.DataFrame
        Source dataframe.
    x : str
        Column for the horizontal (value) axis.
    y : str
        Column for the vertical (category) axis.
    title : str
        Plot title.
    xlabel : str
        X-axis label.
    ylabel : str
        Y-axis label.
    figsize : tuple[int, int]
        Figure size (width, height).
    """
    fig, ax = plt.subplots(figsize=figsize)
    order = data.groupby(y)[x].mean().sort_values(ascending=True).index
    sns.barplot(data=data, x=x, y=y, order=order, ax=ax, errorbar="sd")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    plt.tight_layout()
    plt.show()
