"""Dataframe Type Validation"""

from typing import Any

import pandas as pd
from pandas.api.types import infer_dtype


class PandasTypeValidation:
    """Pandas type validation."""

    @staticmethod
    def infer_dtype(series: pd.Series) -> str:
        """Infer the dtype of a pandas Series."""
        return infer_dtype(series)

    @staticmethod
    def assert_is_dataframe(df: Any) -> None:
        """Assert that the input is a pandas DataFrame."""
        if not isinstance(df, pd.DataFrame):
            raise ValueError(f"{type(df)=}")

    @staticmethod
    def assert_is_series(series: pd.Series) -> None:
        """Assert that the input is a pandas Series."""
        if not isinstance(series, pd.Series):
            raise ValueError(f"{type(series)=}")

    @staticmethod
    def assert_is_index(index: pd.Index) -> None:
        """Assert that the input is a pandas Index."""
        if not isinstance(index, pd.Index):
            raise ValueError(f"{type(index)=}")
