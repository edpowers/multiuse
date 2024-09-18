"""Parquet IO."""

import json
import logging
from pathlib import Path
from typing import Any, Union

import pandas as pd
import pyarrow


class ParquetIO:
    """Reading/writing to parquet files.

    This class is used to read in parquet files and write to parquet files.

    Methods
    -------
    write_to_parquet(df: pd.DataFrame, fpath: Path) -> None
        Write to the parquet file specified.

    Example
    -------
    >>> ParquetIO.write_to_parquet(df, fpath)
    """

    @classmethod
    def write_to_parquet(
        cls,
        df: pd.DataFrame,
        fpath: Path,
        convert_to_json: bool = False,
        return_combined_df: bool = False,
        overwrite: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Write to the parquet file specified.

        Validates the fpath after writing to make sure that it exists.
        Will auto-combine with the existing parquet file if it exists.
        """
        instance = cls()
        final = instance._write_to_parquet(
            df,
            fpath,
            convert_to_json=convert_to_json,
            return_combined_df=return_combined_df,
            overwrite=overwrite,
        )

        if return_combined_df and isinstance(final, pd.DataFrame):
            return final

        return None

    def _write_to_parquet(
        self,
        df: pd.DataFrame,
        fpath: Path,
        convert_to_json: bool = False,
        return_combined_df: bool = False,
        overwrite: bool = False,
    ) -> Union[None, pd.DataFrame]:
        """Write the dataframe to parquet."""
        # Make the corresponding directory if it does not exist.
        # and set the write permissions for access.
        self._create_parent_struct_make_file_permissions(fpath)

        # Convert categorical columns to string
        df = self._convert_categorical_to_string(df)
        # Drop duplicated columns
        df = self.drop_duplicated_columns(df)

        if convert_to_json:
            df = convert_dict_columns_to_json(df)

        # If the old fpath exists, read it and concatenate the new df.
        if fpath.exists() and not overwrite:
            df_old = self.read_parquet(fpath, return_empty_if_error_or_missing=True)
            df = self._concatenate_and_drop_duplicates(df_old, df)
        elif fpath.exists() and overwrite:
            # Then remove the file.
            fpath.unlink(missing_ok=True)

        df.to_parquet(fpath)

        self._validate_fpath_after_write(fpath)

        return df if return_combined_df else None

    def _convert_categorical_to_string(self, df: pd.DataFrame) -> pd.DataFrame:
        """Convert categorical columns to string type.

        This is mainly because pyarrow doesn't understand categorical types.
        And when we read back in, it will throw an error.
        """
        for col in df.select_dtypes(include=["category"]).columns:
            df[col] = df[col].astype(str)
        return df

    def _create_parent_struct_make_file_permissions(self, fpath: Path) -> None:
        """Create the parent directory structure and set the file permissions."""
        if not fpath.exists():
            fpath.parent.mkdir(parents=True, exist_ok=True, mode=0o755)

    def drop_duplicated_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """Drop duplicated columns from the DataFrame.

        This function keeps the first occurrence of each duplicated column
        and removes subsequent occurrences.

        Args:
            df (pd.DataFrame): Input DataFrame

        Returns:
            pd.DataFrame: DataFrame with duplicated columns removed
        """
        return df.loc[:, ~df.columns.duplicated()]

    @staticmethod
    def read_parquet(
        fpath: Path,
        convert_from_json: bool = False,
        return_empty_if_error_or_missing: bool = False,
        columns_for_fake_df: list = [],
    ) -> pd.DataFrame:
        """Read in the parquet file.

        If the file is missing or there is an error, return an empty dataframe.
        and remove the filepath if there is an error.

        Args
        ----
        fpath (Path):
            The file path to read in.

        return_empty_if_error_or_missing (bool, default = False):
            If True, return an empty dataframe if the file is missing.

        """
        df_to_return = pd.DataFrame()

        try:
            df_to_return = pd.read_parquet(fpath)
        except (pyarrow.lib.ArrowIOError, pyarrow.ArrowInvalid, OSError) as aie:
            logging.error(f"ArrowIOError: removing corrupted fpath - {aie}")
            fpath.unlink(missing_ok=True)

        if return_empty_if_error_or_missing:
            df_to_return = pd.DataFrame(columns=columns_for_fake_df)

        if convert_from_json and not df_to_return.empty:
            df_to_return = convert_from_json_to_dict(df_to_return)

        return df_to_return

    def _concatenate_and_drop_duplicates(
        self, df_old: pd.DataFrame, df_new: pd.DataFrame
    ) -> pd.DataFrame:
        """Concatenate and drop duplicates."""
        if df_old.empty:
            return df_new

        return pd.concat([df_old, df_new], ignore_index=True).drop_duplicates(
            ignore_index=True
        )

    def _validate_fpath_after_write(self, fpath: Path) -> None:
        """Validate the written parquet file."""
        if not fpath.exists():
            raise FileNotFoundError(f"Failed to write to {fpath=}")

        logging.info(f"Successfully wrote to {fpath=}")


def load_from_json(x: Union[str, int, float]) -> Union[dict, str, int, float]:
    """Load from json."""
    if not isinstance(x, str):
        return x

    if "{" not in x and "}" not in x:
        return x
    try:
        return json.loads(x)
    except json.JSONDecodeError:
        return x


def convert_to_json(x: Any) -> Union[str, int, float]:
    """Convert various data types to JSON-compatible format."""
    if isinstance(x, (str, int, float)):
        return x

    if isinstance(x, (dict, list)):
        try:
            return json.dumps(x, default=str)
        except TypeError:
            pass

    # Handle other types
    try:
        # Try to convert to dict first (for custom objects)
        return json.dumps(vars(x), default=str)
    except TypeError:
        # If that fails, convert to string
        return str(x)

    # Handle other types
    try:
        # Try to convert to dict first (for custom objects)
        return json.dumps(vars(x), default=str)
    except TypeError:
        # If that fails, convert to string
        return str(x)


def convert_from_json_to_dict(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any columns that are json to dict."""
    for col in df.columns:
        if hasattr(df[col], "str"):
            # If the column contains {} then convert to json.
            if df[col].str.contains(r"{").any():
                df[col] = df[col].apply(lambda x: load_from_json(x))
        elif df[col].dtype == "object":
            df[col] = df[col].astype(str).apply(lambda x: load_from_json(x))

    return df


def convert_dict_columns_to_json(df: pd.DataFrame) -> pd.DataFrame:
    """Convert any columns that are dicts to json."""
    df = df.copy()

    for col in df.columns:
        # if trustradius_all[col]
        # if any of the rows are a dict, then convert to json.
        if df[col].apply(lambda x: isinstance(x, dict)).any():
            df[col] = df[col].apply(lambda x: convert_to_json(x))

    return df
