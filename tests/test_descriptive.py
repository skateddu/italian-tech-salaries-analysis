"""Tests for descriptive statistics functions."""

import pandas as pd

from src.data.descriptive import describe_complete, describe_groups


class TestDescribeComplete:
    """Tests for describe_complete."""

    def test_returns_dataframe(self) -> None:
        df = pd.DataFrame({"salary": [30_000, 40_000, 50_000, 60_000]})
        result = describe_complete(df, column="salary")
        assert isinstance(result, pd.DataFrame)

    def test_includes_standard_stats(self) -> None:
        df = pd.DataFrame({"salary": [30_000, 40_000, 50_000, 60_000]})
        result = describe_complete(df, column="salary")
        expected_columns = {"count", "mean", "std", "min", "25%", "50%", "75%", "max"}
        assert expected_columns.issubset(result.columns)

    def test_includes_skew_and_kurtosis(self) -> None:
        df = pd.DataFrame({"salary": [30_000, 40_000, 50_000, 60_000]})
        result = describe_complete(df, column="salary")
        assert "skew" in result.columns
        assert "kurtosis" in result.columns

    def test_values_are_rounded(self) -> None:
        df = pd.DataFrame({"salary": [33_333, 66_667]})
        result = describe_complete(df, column="salary")
        mean_val = result["mean"].iloc[0]
        # Rounded to 2 decimals
        assert mean_val == round(mean_val, 2)


class TestDescribeGroups:
    """Tests for describe_groups."""

    def test_returns_dataframe(self) -> None:
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
            }
        )
        result = describe_groups(df, group_column="group", value_column="value")
        assert isinstance(result, pd.DataFrame)

    def test_columns_present(self) -> None:
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "value": [10, 20, 30, 40],
            }
        )
        result = describe_groups(df, group_column="group", value_column="value")
        expected = {"n", "mean", "std", "min", "median", "max"}
        assert expected == set(result.columns)

    def test_group_count(self) -> None:
        df = pd.DataFrame(
            {
                "group": ["A", "A", "A", "B", "B"],
                "value": [10, 20, 30, 40, 50],
            }
        )
        result = describe_groups(df, group_column="group", value_column="value")
        assert result.loc["A", "n"] == 3
        assert result.loc["B", "n"] == 2

    def test_group_mean_correct(self) -> None:
        df = pd.DataFrame(
            {
                "group": ["A", "A", "B", "B"],
                "value": [10.0, 30.0, 100.0, 200.0],
            }
        )
        result = describe_groups(df, group_column="group", value_column="value")
        assert result.loc["A", "mean"] == 20.0
        assert result.loc["B", "mean"] == 150.0
