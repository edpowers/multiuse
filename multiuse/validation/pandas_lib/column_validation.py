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
        assert ColumnValidation.validate_column_is_datetime(
            df, col_name
        ), f"{type(df[col_name])=}"

    @staticmethod
    def assert_column_is_integer(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is an integer."""
        assert df[col_name].dtype == "int64", f"{df[col_name].dtype=}"

    @staticmethod
    def assert_not_all_null(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is not all null."""
        assert not df[col_name].isnull().all(), f"{col_name=} is all null."

    @staticmethod
    def assert_not_any_null(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is not all null."""
        assert not df[col_name].isnull().any(), f"{col_name=} is all null."

    @staticmethod
    def assert_not_all_same_value(df: pd.DataFrame, col_name: str) -> None:
        """Assert that a column is not all the same value."""
        assert df[col_name].nunique() != 1, f"{col_name:=} is all the same value."
