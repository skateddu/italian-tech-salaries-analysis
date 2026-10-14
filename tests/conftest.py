"""Shared fixtures for the test suite."""

import numpy as np
import pandas as pd
import pytest


@pytest.fixture()
def salary_df() -> pd.DataFrame:
    """Minimal DataFrame mimicking the raw salary dataset."""
    return pd.DataFrame(
        {
            "province": ["Milano", "Roma", "Napoli", "Torino", "Firenze", "Bari"],
            "monthsOfExperience": [12, 36, 60, 84, 6, 120],
            "salary": [30_000, 35_000, 40_000, 50_000, 28_000, 55_000],
            "submittedAt": [
                "2024-01-15",
                "2024-02-20",
                "2024-03-10",
                "2024-04-05",
                "2024-05-12",
                "2024-06-01",
            ],
        }
    )


@pytest.fixture()
def two_groups() -> tuple[pd.Series, pd.Series]:
    """Two numeric groups with known properties for t-test and effect size."""
    rng = np.random.default_rng(seed=42)
    group1 = pd.Series(rng.normal(loc=50, scale=10, size=100))
    group2 = pd.Series(rng.normal(loc=40, scale=10, size=100))
    return group1, group2


@pytest.fixture()
def three_groups() -> list[pd.Series]:
    """Three numeric groups with known separation for ANOVA."""
    rng = np.random.default_rng(seed=42)
    return [
        pd.Series(rng.normal(loc=30, scale=5, size=50)),
        pd.Series(rng.normal(loc=35, scale=5, size=50)),
        pd.Series(rng.normal(loc=45, scale=5, size=50)),
    ]
