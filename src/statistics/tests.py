"""Wrapper functions for common statistical tests."""

import pandas as pd
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols

from src.statistics.effect_size import cohens_d
from src.statistics.results import AnovaResult, AssumptionResult, Chi2Result, TTestResult


def chi2_test(
    var1: pd.Series, var2: pd.Series, alpha: float = 0.05, label: str = ""
) -> Chi2Result:
    """Run a chi-square test of independence.

    Parameters
    ----------
    var1 : pd.Series
        First categorical variable.
    var2 : pd.Series
        Second categorical variable.
    alpha : float
        Significance level.
    label : str
        Descriptive label for the test.

    Returns
    -------
    Chi2Result
        Dataclass with test statistics and formatted printer.
    """
    observed = pd.crosstab(index=var1, columns=var2)
    chi2, p_value, dof, _ = stats.chi2_contingency(observed)
    return Chi2Result(
        label=label,
        chi2=chi2,
        p_value=p_value,
        dof=dof,
        is_significant=p_value < alpha,
    )


def ttest(
    group1: pd.Series,
    group2: pd.Series,
    label_group1: str,
    label_group2: str,
    alpha: float = 0.05,
) -> TTestResult:
    """Run an independent two-sample t-test.

    Parameters
    ----------
    group1 : pd.Series
        Observations for the first group.
    group2 : pd.Series
        Observations for the second group.
    label_group1 : str
        Display label for the first group.
    label_group2 : str
        Display label for the second group.
    alpha : float
        Significance level.

    Returns
    -------
    TTestResult
        Dataclass with test statistics and formatted printer.
    """
    t_statistic, p_value = stats.ttest_ind(a=group1, b=group2)
    d = cohens_d(group1, group2)
    return TTestResult(
        label_group1=label_group1,
        label_group2=label_group2,
        mean_group1=group1.mean(),
        mean_group2=group2.mean(),
        n_group1=len(group1),
        n_group2=len(group2),
        t_statistic=t_statistic,
        p_value=p_value,
        cohens_d=d,
        is_significant=p_value < alpha,
    )


def oneway_anova(
    groups: list[pd.Series],
    label: str,
    alpha: float = 0.05,
    non_parametric: bool = False,
) -> AnovaResult:
    """Run a one-way ANOVA or Kruskal-Wallis test.

    Parameters
    ----------
    groups : list[pd.Series]
        List of observation series, one per group.
    label : str
        Descriptive label for the comparison.
    alpha : float
        Significance level.
    non_parametric : bool
        If True, use Kruskal-Wallis instead of ANOVA.

    Returns
    -------
    AnovaResult
        Dataclass with test statistics and formatted printer.
    """
    if non_parametric:
        statistic, p_value = stats.kruskal(*groups)
    else:
        statistic, p_value = stats.f_oneway(*groups)
    return AnovaResult(
        label=label,
        statistic=statistic,
        p_value=p_value,
        is_significant=p_value < alpha,
        is_non_parametric=non_parametric,
    )


def check_anova_assumptions(
    groups: list[pd.Series],
    labels: list[str],
    label: str,
    alpha: float = 0.05,
    max_shapiro_n: int = 5000,
) -> AssumptionResult:
    """Check normality and homoscedasticity assumptions for ANOVA.

    Parameters
    ----------
    groups : list[pd.Series]
        List of observation series, one per group.
    labels : list[str]
        Display labels for each group.
    label : str
        Descriptive label for the overall comparison.
    alpha : float
        Significance level for assumption tests.
    max_shapiro_n : int
        Maximum sample size for Shapiro-Wilk. Larger groups are subsampled
        to avoid oversensitivity with very large n.

    Returns
    -------
    AssumptionResult
        Dataclass with assumption test results and recommendation.
    """
    normality_tests = []
    for group, group_label in zip(groups, labels):
        sample = (
            group.sample(n=max_shapiro_n, random_state=42) if len(group) > max_shapiro_n else group
        )
        statistic, p_value = stats.shapiro(sample)
        normality_tests.append(
            {
                "group": group_label,
                "statistic": statistic,
                "p_value": p_value,
                "is_normal": p_value >= alpha,
                "n": len(group),
            }
        )

    levene_statistic, levene_p_value = stats.levene(*groups)
    is_homoscedastic = levene_p_value >= alpha

    is_all_normal = all(t["is_normal"] for t in normality_tests)
    is_all_large_n = all(t["n"] >= 30 for t in normality_tests)

    if is_all_normal and is_homoscedastic:
        recommendation = "Assunzioni rispettate. ANOVA parametrica appropriata."
    elif is_all_normal and not is_homoscedastic:
        recommendation = (
            "Normalità rispettata ma varianze non omogenee. "
            "Considerare Welch ANOVA o Kruskal-Wallis."
        )
    elif not is_all_normal and is_all_large_n:
        recommendation = (
            "Normalità violata, ma tutti i gruppi hanno n >= 30: "
            "l'ANOVA è robusta per il Teorema del Limite Centrale."
        )
    else:
        recommendation = (
            "Normalità violata con campioni piccoli. "
            "Usare Kruskal-Wallis (alternativa non parametrica)."
        )

    return AssumptionResult(
        label=label,
        normality_tests=normality_tests,
        levene_statistic=levene_statistic,
        levene_p_value=levene_p_value,
        is_homoscedastic=is_homoscedastic,
        recommendation=recommendation,
    )


def factorial_anova(
    data: pd.DataFrame, target: str, factors: list[str], anova_type: int = 2
) -> pd.DataFrame:
    """Run a factorial ANOVA using OLS.

    Parameters
    ----------
    data : pd.DataFrame
        Input data containing target and factor columns.
    target : str
        Name of the continuous dependent variable.
    factors : list[str]
        Names of the categorical independent variables.
    anova_type : int
        Type of sum of squares (1, 2, or 3).

    Returns
    -------
    pd.DataFrame
        ANOVA table produced by statsmodels.
    """
    terms = " * ".join(f"C({factor})" for factor in factors)
    formula = f"{target} ~ {terms}"
    data = data[[target, *factors]].dropna()
    model = ols(formula, data).fit()
    return sm.stats.anova_lm(model, typ=anova_type).round(decimals=2)
