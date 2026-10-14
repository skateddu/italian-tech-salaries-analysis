"""Data loading, processing, and geographic mapping utilities."""

from salaries_analysis.data.dataset import load_salaries
from salaries_analysis.data.descriptive import describe_complete, describe_groups
from salaries_analysis.data.features import feature_engineering

__all__ = [
    "feature_engineering",
    "describe_complete",
    "describe_groups",
    "load_salaries",
]
