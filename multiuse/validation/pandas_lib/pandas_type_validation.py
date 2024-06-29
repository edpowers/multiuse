"""Dataframe Type Validation"""

import pandas as pd
from typing import Any


class PandasTypeValidation:
    """Pandas type validation."""

    @staticmethod
    def assert_is_dataframe(df: Any) -> None:
        """Assert that the input is a pandas DataFrame."""
        assert isinstance(df, pd.DataFrame), f"{type(df)=}"

    @staticmethod
    def assert_is_series(series: pd.Series) -> None:
        """Assert that the input is a pandas Series."""
        assert isinstance(series, pd.Series), f"{type(series)=}"

    @staticmethod
    def assert_is_index(index: pd.Index) -> None:
        """Assert that the input is a pandas Index."""
        assert isinstance(index, pd.Index), f"{type(index)=}"
