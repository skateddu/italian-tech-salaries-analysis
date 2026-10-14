"""Statistical analysis tools for salary data."""

from salaries_analysis.statistics.effect_size import cohens_d
from salaries_analysis.statistics.results import (
    AnovaResult,
    AssumptionResult,
    Chi2Result,
    TTestResult,
)
from salaries_analysis.statistics.tests import (
    adjust_pvalues,
    check_anova_assumptions,
    chi2_test,
    factorial_anova,
    oneway_anova,
    ttest,
)

__all__ = [
    "adjust_pvalues",
    "AnovaResult",
    "AssumptionResult",
    "Chi2Result",
    "TTestResult",
    "check_anova_assumptions",
    "chi2_test",
    "cohens_d",
    "factorial_anova",
    "oneway_anova",
    "ttest",
]
