"""Feature engineering transformations for salary data."""

import numpy as np
import pandas as pd

from src.data.geo_mapper import PROVINCE_MAPPER, REGIONS_MAPPER

EXPERIENCE_JUNIOR_THRESHOLD = 2
EXPERIENCE_MIDDLE_THRESHOLD = 5


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Perform feature engineering for geographic, experience, and temporal features.

    Parameters
    ----------
    df : pd.DataFrame
        Raw salary DataFrame.

    Returns
    -------
    pd.DataFrame
        DataFrame with additional feature columns.
    """
    df = df.copy()
    df["region"] = df["province"].map(PROVINCE_MAPPER)
    df["macro_region"] = df["region"].map(REGIONS_MAPPER)
    df["yearsOfExperience"] = df["monthsOfExperience"] // 12
    df["experience"] = np.select(
        condlist=[
            df["yearsOfExperience"] <= EXPERIENCE_JUNIOR_THRESHOLD,
            df["yearsOfExperience"] <= EXPERIENCE_MIDDLE_THRESHOLD,
        ],
        choicelist=["junior", "middle"],
        default="senior",
    )
    df["yearSubmittedAt"] = pd.to_datetime(df["submittedAt"], format="mixed").dt.year
    return df
