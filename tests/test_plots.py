"""Tests for plotting helpers."""

import matplotlib

matplotlib.use("Agg")

from matplotlib.axes import Axes
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from salaries_analysis.visualization import (
    STYLE_PATH,
    plot_heatmap,
    plot_horizontal_barplot,
    plot_salary_boxplot,
)


@pytest.fixture()
def plot_df() -> pd.DataFrame:
    """Small DataFrame with a categorical and a numeric column."""
    return pd.DataFrame(
        {
            "group": ["a", "a", "b", "b", "c", "c"],
            "salary": [30_000, 32_000, 40_000, 42_000, 50_000, 55_000],
        }
    )


@pytest.fixture(autouse=True)
def close_figures():
    """Close all figures after each test."""
    yield
    plt.close("all")


class TestStylePath:
    """Tests for the bundled Matplotlib style."""

    def test_style_file_exists(self) -> None:
        assert STYLE_PATH.is_file()

    def test_style_is_loadable(self) -> None:
        plt.style.use(STYLE_PATH)


class TestPlotSalaryBoxplot:
    """Tests for plot_salary_boxplot."""

    def test_returns_axes_with_labels(self, plot_df: pd.DataFrame) -> None:
        ax = plot_salary_boxplot(plot_df, x="group", title="T", xlabel="X")
        assert isinstance(ax, Axes)
        assert ax.get_title() == "T"
        assert ax.get_xlabel() == "X"
        assert ax.get_ylabel() == "Salary (€)"

    def test_respects_order_and_ylim(self, plot_df: pd.DataFrame) -> None:
        ax = plot_salary_boxplot(plot_df, x="group", order=["c", "b", "a"], ylim=(0, 60_000))
        assert [t.get_text() for t in ax.get_xticklabels()] == ["c", "b", "a"]
        assert ax.get_ylim() == (0, 60_000)

    def test_draws_on_given_axes(self, plot_df: pd.DataFrame) -> None:
        _, given = plt.subplots()
        ax = plot_salary_boxplot(plot_df, x="group", ax=given)
        assert ax is given


class TestPlotHeatmap:
    """Tests for plot_heatmap."""

    def test_annotates_every_cell(self) -> None:
        data = pd.DataFrame(
            [[1_000, 2_000], [3_000, 4_000]], index=["r1", "r2"], columns=["c1", "c2"]
        )
        ax = plot_heatmap(data, title="Heat")
        assert ax.get_title() == "Heat"
        assert [t.get_text() for t in ax.texts] == ["1,000", "2,000", "3,000", "4,000"]


class TestPlotHorizontalBarplot:
    """Tests for plot_horizontal_barplot."""

    def test_categories_sorted_by_mean(self, plot_df: pd.DataFrame) -> None:
        ax = plot_horizontal_barplot(plot_df, x="salary", y="group")
        assert [t.get_text() for t in ax.get_yticklabels()] == ["a", "b", "c"]
