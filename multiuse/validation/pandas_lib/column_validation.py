"""Pandas column validation."""

import pandas as pd


class ColumnValidation:
    """Pandas column validation."""

    @staticmethod
    def validate_column_is_datetime(df: pd.DataFrame, col_name: str) -> bool:
        """Validate that a column is a datetime."""
        return hasattr(df[col_name], "dt")

    @staticmethod
    def assert_column_is_datetime(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is a datetime."""
        if not ColumnValidation.validate_column_is_datetime(df, col_name):
            raise ValueError(f"{type(df[col_name])=}")

    @staticmethod
    def assert_column_is_integer(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is an integer."""
        if df[col_name].dtype != "int64":
            raise ValueError(f"{df[col_name].dtype=}")

    @staticmethod
    def assert_not_all_null(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is not all null."""
        if not df[col_name].isnull().all():
            raise ValueError(f"{col_name=} is all null.")

    @staticmethod
    def assert_not_any_null(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is not all null."""
        if df[col_name].isnull().any():
            raise ValueError(f"{col_name=} has null values.")

    @staticmethod
    def assert_not_all_same_value(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is not all the same value."""
        if df[col_name].nunique() != 1:
            raise ValueError(f"{col_name:=} is all the same value.")
