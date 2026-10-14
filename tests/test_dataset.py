"""Tests for the dataset loader."""

from unittest.mock import MagicMock, patch

import pandas as pd

from salaries_analysis.data import dataset


class TestLoadSalaries:
    """Tests for load_salaries."""

    def test_uses_pinned_revision_by_default(self) -> None:
        fake = {"train": MagicMock()}
        fake["train"].to_pandas.return_value = pd.DataFrame({"salary": [1]})
        with patch.object(dataset, "load_dataset", return_value=fake) as mock_load:
            df = dataset.load_salaries()
        mock_load.assert_called_once_with(dataset.DATASET_PATH, revision=dataset.DATASET_REVISION)
        assert list(df.columns) == ["salary"]

    def test_revision_none_loads_latest(self) -> None:
        fake = {"train": MagicMock()}
        with patch.object(dataset, "load_dataset", return_value=fake) as mock_load:
            dataset.load_salaries(revision=None)
        mock_load.assert_called_once_with(dataset.DATASET_PATH, revision=None)
