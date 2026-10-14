"""Tests for statistical test wrapper functions."""

import numpy as np
import pandas as pd

from src.statistics.results import AnovaResult, AssumptionResult, Chi2Result, TTestResult
from src.statistics.tests import (
    check_anova_assumptions,
    chi2_test,
    factorial_anova,
    oneway_anova,
    ttest,
)


class TestTtest:
    """Tests for the ttest wrapper."""

    def test_returns_ttest_result(self, two_groups: tuple[pd.Series, pd.Series]) -> None:
        group1, group2 = two_groups
        result = ttest(group1, group2, "G1", "G2")
        assert isinstance(result, TTestResult)

    def test_significant_difference_detected(
        self, two_groups: tuple[pd.Series, pd.Series]
    ) -> None:
        group1, group2 = two_groups
        result = ttest(group1, group2, "G1", "G2")
        assert result.is_significant
        assert result.p_value < 0.05

    def test_no_difference_not_significant(self) -> None:
        rng = np.random.default_rng(seed=42)
        same = pd.Series(rng.normal(loc=50, scale=10, size=100))
        result = ttest(same, same, "A", "A_copy")
        assert not result.is_significant
        assert result.p_value == 1.0

    def test_group_metadata_correct(self) -> None:
        g1 = pd.Series([10.0, 20.0, 30.0])
        g2 = pd.Series([40.0, 50.0])
        result = ttest(g1, g2, "Small", "Big")
        assert result.label_group1 == "Small"
        assert result.label_group2 == "Big"
        assert result.n_group1 == 3
        assert result.n_group2 == 2
        assert abs(result.mean_group1 - 20.0) < 1e-10
        assert abs(result.mean_group2 - 45.0) < 1e-10

    def test_custom_alpha(self, two_groups: tuple[pd.Series, pd.Series]) -> None:
        group1, group2 = two_groups
        result = ttest(group1, group2, "G1", "G2", alpha=1e-20)
        assert result.is_significant == (result.p_value < 1e-20)


class TestOnewayAnova:
    """Tests for the oneway_anova wrapper."""

    def test_returns_anova_result(self, three_groups: list[pd.Series]) -> None:
        result = oneway_anova(three_groups, label="Test")
        assert isinstance(result, AnovaResult)

    def test_significant_groups_detected(self, three_groups: list[pd.Series]) -> None:
        result = oneway_anova(three_groups, label="Test")
        assert result.is_significant
        assert not result.is_non_parametric

    def test_non_parametric_mode(self, three_groups: list[pd.Series]) -> None:
        result = oneway_anova(three_groups, label="Test", non_parametric=True)
        assert result.is_significant
        assert result.is_non_parametric

    def test_identical_groups_not_significant(self) -> None:
        rng = np.random.default_rng(seed=42)
        same_data = pd.Series(rng.normal(loc=50, scale=10, size=50))
        result = oneway_anova([same_data, same_data, same_data], label="Same")
        assert not result.is_significant


class TestChi2Test:
    """Tests for the chi2_test wrapper."""

    def test_returns_chi2_result(self) -> None:
        var1 = pd.Series(["A", "A", "B", "B", "A", "B"] * 20)
        var2 = pd.Series(["X", "Y", "X", "Y", "X", "Y"] * 20)
        result = chi2_test(var1, var2, label="Test")
        assert isinstance(result, Chi2Result)

    def test_associated_variables_significant(self) -> None:
        var1 = pd.Series(["A"] * 100 + ["B"] * 100)
        var2 = pd.Series(["X"] * 100 + ["Y"] * 100)
        result = chi2_test(var1, var2, label="Associated")
        assert result.is_significant

    def test_independent_variables_not_significant(self) -> None:
        rng = np.random.default_rng(seed=42)
        var1 = pd.Series(rng.choice(["A", "B"], size=200))
        var2 = pd.Series(rng.choice(["X", "Y"], size=200))
        result = chi2_test(var1, var2, label="Independent")
        assert not result.is_significant

    def test_degrees_of_freedom(self) -> None:
        var1 = pd.Series(["A", "B", "C"] * 30)
        var2 = pd.Series(["X", "Y"] * 45)
        result = chi2_test(var1, var2, label="DOF")
        assert result.dof == 2


class TestCheckAnovaAssumptions:
    """Tests for ANOVA assumption checking."""

    def test_returns_assumption_result(self, three_groups: list[pd.Series]) -> None:
        result = check_anova_assumptions(
            groups=three_groups,
            labels=["A", "B", "C"],
            label="Test",
        )
        assert isinstance(result, AssumptionResult)

    def test_normal_homoscedastic_recommends_anova(self) -> None:
        rng = np.random.default_rng(seed=42)
        groups = [pd.Series(rng.normal(loc=i, scale=5, size=30)) for i in range(3)]
        result = check_anova_assumptions(
            groups=groups,
            labels=["A", "B", "C"],
            label="Normal",
        )
        if result.normality_tests[0]["is_normal"] and result.is_homoscedastic:
            assert "ANOVA parametrica" in result.recommendation

    def test_small_non_normal_recommends_kruskal(self) -> None:
        rng = np.random.default_rng(seed=42)
        groups = [pd.Series(rng.exponential(scale=10, size=15)) for _ in range(3)]
        result = check_anova_assumptions(
            groups=groups,
            labels=["A", "B", "C"],
            label="Skewed",
        )
        is_all_normal = all(t["is_normal"] for t in result.normality_tests)
        is_all_large = all(t["n"] >= 30 for t in result.normality_tests)
        if not is_all_normal and not is_all_large:
            assert "Kruskal-Wallis" in result.recommendation

    def test_large_n_recommends_clt_robustness(self) -> None:
        rng = np.random.default_rng(seed=42)
        groups = [pd.Series(rng.exponential(scale=10, size=500)) for _ in range(3)]
        result = check_anova_assumptions(
            groups=groups,
            labels=["A", "B", "C"],
            label="Large",
        )
        is_all_normal = all(t["is_normal"] for t in result.normality_tests)
        is_all_large = all(t["n"] >= 30 for t in result.normality_tests)
        if not is_all_normal and is_all_large:
            assert "Teorema del Limite Centrale" in result.recommendation

    def test_normality_tests_count_matches_groups(self) -> None:
        rng = np.random.default_rng(seed=42)
        groups = [pd.Series(rng.normal(size=50)) for _ in range(4)]
        result = check_anova_assumptions(
            groups=groups,
            labels=["A", "B", "C", "D"],
            label="Four",
        )
        assert len(result.normality_tests) == 4

    def test_subsampling_for_large_groups(self) -> None:
        rng = np.random.default_rng(seed=42)
        large_group = pd.Series(rng.normal(size=10_000))
        result = check_anova_assumptions(
            groups=[large_group, large_group],
            labels=["A", "B"],
            label="Large",
            max_shapiro_n=100,
        )
        assert len(result.normality_tests) == 2
        assert result.normality_tests[0]["n"] == 10_000


class TestFactorialAnova:
    """Tests for factorial ANOVA."""

    def test_returns_dataframe(self) -> None:
        rng = np.random.default_rng(seed=42)
        df = pd.DataFrame(
            {
                "salary": rng.normal(loc=40000, scale=10000, size=120),
                "experience": (["junior"] * 40 + ["middle"] * 40 + ["senior"] * 40),
                "region": (["Nord", "Sud"] * 60),
            }
        )
        result = factorial_anova(df, target="salary", factors=["experience", "region"])
        assert isinstance(result, pd.DataFrame)
        assert "PR(>F)" in result.columns

    def test_significant_factor_detected(self) -> None:
        rng = np.random.default_rng(seed=42)
        n = 60
        experience = ["junior"] * n + ["senior"] * n
        salary = np.concatenate(
            [
                rng.normal(loc=30000, scale=5000, size=n),
                rng.normal(loc=50000, scale=5000, size=n),
            ]
        )
        region = ["Nord", "Sud"] * n
        df = pd.DataFrame(
            {
                "salary": salary,
                "experience": experience,
                "region": region,
            }
        )
        result = factorial_anova(df, target="salary", factors=["experience", "region"])
        experience_p = result.loc["C(experience)", "PR(>F)"]
        assert experience_p < 0.05
