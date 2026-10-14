"""Tests for statistical result dataclasses."""

from salaries_analysis.statistics.results import (
    AnovaResult,
    AssumptionResult,
    Chi2Result,
    TTestResult,
)


class TestTTestResult:
    """Tests for TTestResult dataclass."""

    def test_print_report_runs(self, capsys: object) -> None:
        result = TTestResult(
            label_group1="A",
            label_group2="B",
            mean_group1=50.0,
            mean_group2=40.0,
            n_group1=100,
            n_group2=100,
            t_statistic=3.5,
            p_value=0.001,
            cohens_d=0.7,
            is_significant=True,
        )
        result.print_report()
        captured = capsys.readouterr()
        assert "A vs B" in captured.out
        assert "3.50" in captured.out


class TestAnovaResult:
    """Tests for AnovaResult dataclass."""

    def test_print_report_parametric(self, capsys: object) -> None:
        result = AnovaResult(
            label="Test",
            statistic=15.0,
            p_value=0.001,
            is_significant=True,
            is_non_parametric=False,
        )
        result.print_report()
        captured = capsys.readouterr()
        assert "ANOVA" in captured.out
        assert "F-statistic" in captured.out

    def test_print_report_non_parametric(self, capsys: object) -> None:
        result = AnovaResult(
            label="Test",
            statistic=15.0,
            p_value=0.001,
            is_significant=True,
            is_non_parametric=True,
        )
        result.print_report()
        captured = capsys.readouterr()
        assert "Kruskal-Wallis" in captured.out
        assert "H-statistic" in captured.out

    def test_print_report_welch(self, capsys: object) -> None:
        result = AnovaResult(
            label="Test",
            statistic=15.0,
            p_value=0.001,
            is_significant=True,
            is_welch=True,
        )
        result.print_report()
        captured = capsys.readouterr()
        assert "Welch ANOVA" in captured.out
        assert "F-statistic" in captured.out


class TestChi2Result:
    """Tests for Chi2Result dataclass."""

    def test_print_report_runs(self, capsys: object) -> None:
        result = Chi2Result(
            label="X vs Y",
            chi2=12.5,
            p_value=0.002,
            dof=3,
            is_significant=True,
        )
        result.print_report()
        captured = capsys.readouterr()
        assert "X vs Y" in captured.out
        assert "12.50" in captured.out


class TestAssumptionResult:
    """Tests for AssumptionResult dataclass."""

    def test_print_report_runs(self, capsys: object) -> None:
        result = AssumptionResult(
            label="Test",
            normality_tests=[
                {"group": "A", "statistic": 0.98, "p_value": 0.5, "is_normal": True, "n": 50},
                {"group": "B", "statistic": 0.95, "p_value": 0.01, "is_normal": False, "n": 30},
            ],
            levene_statistic=2.5,
            levene_p_value=0.12,
            is_homoscedastic=True,
            recommendation="ANOVA parametrica appropriata.",
        )
        result.print_report()
        captured = capsys.readouterr()
        assert "normale" in captured.out
        assert "non normale" in captured.out
        assert "omogenee" in captured.out
