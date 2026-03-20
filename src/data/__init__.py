"""Data loading, processing, and geographic mapping utilities."""

from src.data.dataset import load_salaries
from src.data.descriptive import describe_complete, describe_groups
from src.data.features import feature_engineering

__all__ = [
    "feature_engineering",
    "describe_complete",
    "describe_groups",
    "load_salaries",
]
