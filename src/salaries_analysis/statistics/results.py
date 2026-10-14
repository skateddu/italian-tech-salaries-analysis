"""Dataclasses for presenting statistical test results."""

from dataclasses import dataclass, field


@dataclass
class TTestResult:
    """Container for t-test results."""

    label_group1: str
    label_group2: str
    mean_group1: float
    mean_group2: float
    n_group1: int
    n_group2: int
    t_statistic: float
    p_value: float
    cohens_d: float
    is_significant: bool

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Test t - {self.label_group1} vs {self.label_group2}",
                f"Media {self.label_group1}: €{self.mean_group1:.2f} | "
                f"Media {self.label_group2}: €{self.mean_group2:.2f}",
                f"t-statistic: {self.t_statistic:.2f}, P-value: {self.p_value:.2e}",
                f"Effect size (Cohen's d): {self.cohens_d:.2f}",
            ]
        )

    def print_report(self) -> None:
        """Print a formatted summary of the t-test."""
        print(self)


@dataclass
class AnovaResult:
    """Container for one-way ANOVA, Welch ANOVA or Kruskal-Wallis results."""

    label: str
    statistic: float
    p_value: float
    is_significant: bool
    is_non_parametric: bool = False
    is_welch: bool = False

    def __str__(self) -> str:
        if self.is_non_parametric:
            test_name = "Kruskal-Wallis"
        elif self.is_welch:
            test_name = "Welch ANOVA"
        else:
            test_name = "ANOVA"
        stat_name = "H-statistic" if self.is_non_parametric else "F-statistic"

        return "\n".join(
            [
                f"{test_name} - {self.label}",
                f"{stat_name}: {self.statistic:.2f}",
                f"P-value: {self.p_value:.2e}",
                f"Differenze significative (α=0.05): {self.is_significant}",
            ]
        )

    def print_report(self) -> None:
        """Print a formatted summary of the ANOVA / Welch / Kruskal-Wallis test."""
        print(self)


@dataclass
class Chi2Result:
    """Container for chi-square test results."""

    label: str
    chi2: float
    p_value: float
    dof: int
    is_significant: bool

    def __str__(self) -> str:
        return "\n".join(
            [
                f"Chi² - {self.label}",
                f"Chi² = {self.chi2:.2f}, P-value = {self.p_value:.2e}, df = {self.dof}",
                f"Associazione significativa: {self.is_significant}",
            ]
        )

    def print_report(self) -> None:
        """Print a formatted summary of the chi-square test."""
        print(self)


@dataclass
class AssumptionResult:
    """Container for ANOVA assumption verification results."""

    label: str
    normality_tests: list[dict] = field(default_factory=list)
    levene_statistic: float = 0.0
    levene_p_value: float = 0.0
    is_homoscedastic: bool = False
    recommendation: str = ""

    def __str__(self) -> str:
        lines = [f"Verifica assunzioni - {self.label}", "Normalità (Shapiro-Wilk, α=0.05):"]
        for test in self.normality_tests:
            status = "normale" if test["is_normal"] else "non normale"
            lines.append(
                f"  {test['group']} (n={test['n']}): "
                f"W={test['statistic']:.4f}, p={test['p_value']:.2e} - {status}"
            )
        status = "omogenee" if self.is_homoscedastic else "non omogenee"
        lines += [
            "Omoschedasticità (Levene, α=0.05):",
            f"  F={self.levene_statistic:.2f}, p={self.levene_p_value:.2e} - varianze {status}",
            f"Raccomandazione: {self.recommendation}",
        ]
        return "\n".join(lines)

    def print_report(self) -> None:
        """Print a formatted summary of assumption checks."""
        print(self)
