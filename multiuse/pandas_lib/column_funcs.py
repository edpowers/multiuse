"""Column functions for pandas DataFrames."""

import humps
import pandas as pd


class ColumnFuncs:
    """Staticmethods helpers for pandas DataFrames."""

    @staticmethod
    def standardize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Standardize column names to lowercase and replace spaces with underscores."""
        return df.rename(
            columns=lambda x: humps.decamelize(x).lower().strip().replace(" ", "_")
        )

    @staticmethod
    def drop_duplicate_cols(df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicate columns.

        Takes a dataframe as input.

        Example
        -------
        >>> df_to_test = pd.DataFrame(columns=["sym_0", "sym_0", "test"])

        >>> df_to_test.columns
        Index(['sym_0', 'sym_0', 'test'], dtype='object')
        >>> df_to_test.values
        array([['a', 'b', 'c']], dtype=object)

        >>> df_new = df_to_test.custom.drop_duplicate_cols()

        >>> df_new.columns
        Index(['sym_0', 'test'], dtype='object')
        >>> df_new.values
        array([['a', 'c']], dtype=object)

        """
        return df.loc[:, ~df.columns.duplicated()]

    @staticmethod
    def drop_single_value_columns(df: pd.DataFrame) -> pd.DataFrame:
        """Drop columns that have only one unique value."""
        for col in df.columns:
            if len(df[col].unique()) == 1:
                df = df.drop(col, axis=1)
        return df
