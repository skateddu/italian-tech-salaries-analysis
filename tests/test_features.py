"""Tests for feature engineering transformations."""

import pandas as pd

from src.data.features import (
    EXPERIENCE_JUNIOR_THRESHOLD,
    EXPERIENCE_MIDDLE_THRESHOLD,
    feature_engineering,
)


class TestFeatureEngineering:
    """Tests for the feature_engineering function."""

    def test_returns_dataframe(self, salary_df: pd.DataFrame) -> None:
        result = feature_engineering(salary_df)
        assert isinstance(result, pd.DataFrame)

    def test_does_not_modify_original(self, salary_df: pd.DataFrame) -> None:
        original_columns = set(salary_df.columns)
        feature_engineering(salary_df)
        assert set(salary_df.columns) == original_columns

    def test_adds_region_column(self, salary_df: pd.DataFrame) -> None:
        result = feature_engineering(salary_df)
        assert "region" in result.columns
        assert result.loc[0, "region"] == "Lombardia"
        assert result.loc[1, "region"] == "Lazio"

    def test_adds_macro_region_column(self, salary_df: pd.DataFrame) -> None:
        result = feature_engineering(salary_df)
        assert "macro_region" in result.columns
        assert result.loc[0, "macro_region"] == "Nord"  # Milano -> Lombardia -> Nord
        assert result.loc[1, "macro_region"] == "Centro"  # Roma -> Lazio -> Centro
        assert result.loc[2, "macro_region"] == "Sud"  # Napoli -> Campania -> Sud

    def test_years_of_experience_calculation(self, salary_df: pd.DataFrame) -> None:
        result = feature_engineering(salary_df)
        assert "yearsOfExperience" in result.columns
        # 12 months = 1 year, 36 months = 3 years
        assert result.loc[0, "yearsOfExperience"] == 1
        assert result.loc[1, "yearsOfExperience"] == 3

    def test_experience_category_junior(self, salary_df: pd.DataFrame) -> None:
        result = feature_engineering(salary_df)
        # 12 months = 1 year <= 2 -> junior
        assert result.loc[0, "experience"] == "junior"
        # 6 months = 0 years <= 2 -> junior
        assert result.loc[4, "experience"] == "junior"

    def test_experience_category_middle(self, salary_df: pd.DataFrame) -> None:
        result = feature_engineering(salary_df)
        # 36 months = 3 years, 2 < 3 <= 5 -> middle
        assert result.loc[1, "experience"] == "middle"
        # 60 months = 5 years, 2 < 5 <= 5 -> middle
        assert result.loc[2, "experience"] == "middle"

    def test_experience_category_senior(self, salary_df: pd.DataFrame) -> None:
        result = feature_engineering(salary_df)
        # 84 months = 7 years > 5 -> senior
        assert result.loc[3, "experience"] == "senior"
        # 120 months = 10 years > 5 -> senior
        assert result.loc[5, "experience"] == "senior"

    def test_year_submitted_at(self, salary_df: pd.DataFrame) -> None:
        result = feature_engineering(salary_df)
        assert "yearSubmittedAt" in result.columns
        assert (result["yearSubmittedAt"] == 2024).all()

    def test_thresholds_are_consistent(self) -> None:
        assert EXPERIENCE_JUNIOR_THRESHOLD < EXPERIENCE_MIDDLE_THRESHOLD
