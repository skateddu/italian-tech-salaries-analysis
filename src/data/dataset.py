"""Loading utilities for the Italian tech salaries dataset."""

import pandas as pd
from datasets import load_dataset


def load_salaries(path: str) -> pd.DataFrame:
    """Load the Italian tech salaries dataset from HuggingFace.

    Parameters
    ----------
    path : str
        The path to the dataset on HuggingFace.

    Returns
    -------
    pd.DataFrame
        The loaded dataset as a pandas DataFrame.
    """
    dataset = load_dataset(path)
    return dataset["train"].to_pandas()
