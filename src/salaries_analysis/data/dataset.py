"""Loading utilities for the Italian tech salaries dataset."""

from datasets import load_dataset
import pandas as pd

DATASET_PATH = "datapizza-ai-lab/salaries"
# Pinned HuggingFace commit (weekly export of 2026-10-11, 26,478 submissions).
DATASET_REVISION = "52ad2e7107a0c51d224ff272475b7b650daabc88"


def load_salaries(
    path: str = DATASET_PATH, revision: str | None = DATASET_REVISION
) -> pd.DataFrame:
    """Load the Italian tech salaries dataset from HuggingFace.

    Parameters
    ----------
    path : str
        The path to the dataset on HuggingFace.
    revision : str or None
        Git revision (commit SHA, branch or tag) to load. Pinned by default
        for reproducibility; pass None to load the latest version.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.
    """
    dataset = load_dataset(path, revision=revision)
    return dataset["train"].to_pandas()
