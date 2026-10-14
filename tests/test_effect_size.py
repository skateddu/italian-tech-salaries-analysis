"""Tests for Cohen's d effect size calculation."""

import numpy as np
import pandas as pd

from src.statistics.effect_size import cohens_d


class TestCohensD:
    """Tests for the cohens_d function."""

    def test_identical_groups_returns_zero(self) -> None:
        data = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        assert cohens_d(data, data) == 0.0

    def test_known_effect_size(self) -> None:
        group1 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
        group2 = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0]) + 15.0

        d = cohens_d(group2, group1)

        # d = mean_diff / pooled_std; pooled_std = std of identical spread
        expected = 15.0 / group1.std()
        assert abs(d - expected) < 1e-10

    def test_sign_reflects_direction(self) -> None:
        group_high = pd.Series([50.0, 60.0, 70.0])
        group_low = pd.Series([20.0, 30.0, 40.0])

        assert cohens_d(group_high, group_low) > 0
        assert cohens_d(group_low, group_high) < 0

    def test_large_effect_size(self, two_groups: tuple[pd.Series, pd.Series]) -> None:
        group1, group2 = two_groups
        d = cohens_d(group1, group2)

        # Groups separated by ~10 with std ~10, expect d ~ 1.0
        assert d > 0.8, f"Expected large effect size (d > 0.8), got {d:.2f}"

    def test_uses_pooled_std(self) -> None:
        group1 = pd.Series([10.0, 20.0, 30.0])
        group2 = pd.Series([100.0, 200.0, 300.0, 400.0])

        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * group1.var() + (n2 - 1) * group2.var()) / (n1 + n2 - 2))
        expected = (group1.mean() - group2.mean()) / pooled_std

        assert abs(cohens_d(group1, group2) - expected) < 1e-10
