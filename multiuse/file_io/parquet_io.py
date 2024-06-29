"""Parquet IO."""

import logging
from pathlib import Path
from typing import Union

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
        cls, df: pd.DataFrame, fpath: Path, return_combined_df: bool = False
    ) -> Union[None, pd.DataFrame]:
        """Write to the parquet file specified.

        Validates the fpath after writing to make sure that it exists.
        Will auto-combine with the existing parquet file if it exists.
        """
        instance = cls()
        final = instance._write_to_parquet(
            df, fpath, return_combined_df=return_combined_df
        )

        if return_combined_df and isinstance(final, pd.DataFrame):
            return final

        return None

    def _write_to_parquet(
        self, df: pd.DataFrame, fpath: Path, return_combined_df: bool = False
    ) -> Union[None, pd.DataFrame]:
        """Write the dataframe to parquet."""
        # Make the corresponding directory if it does not exist.
        # and set the write permissions for access.
        self._create_parent_struct_make_file_permissions(fpath)

        # If the old fpath exists, read it and concatenate the new df.
        if fpath.exists():
            df_old = self.read_parquet(fpath, return_empty_if_error_or_missing=True)
            df = self._concatenate_and_drop_duplicates(df_old, df)

        df.to_parquet(fpath)

        self._validate_fpath_after_write(fpath)

        return df if return_combined_df else None

    def _create_parent_struct_make_file_permissions(self, fpath: Path) -> None:
        """Create the parent directory structure and set the file permissions."""
        if not fpath.exists():
            fpath.parent.mkdir(parents=True, exist_ok=True, mode=0o755)

    @staticmethod
    def read_parquet(
        fpath: Path,
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
        try:
            return pd.read_parquet(fpath)
        except (pyarrow.lib.ArrowIOError, pyarrow.ArrowInvalid, OSError) as aie:
            logging.error(f"ArrowIOError: removing corrupted fpath - {aie}")
            fpath.unlink(missing_ok=True)

        if return_empty_if_error_or_missing:
            return pd.DataFrame(columns=columns_for_fake_df)

        raise FileNotFoundError(f"Failed to read in {fpath=}")

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
