"""Setup the file paths index."""

import os
from pathlib import Path
from typing import ClassVar

import pandas as pd
from rich import print as rprint

from multiuse.file_io.parquet_io import ParquetIO


class BaseFilePaths:
    """Setup the base file paths for easier access.

    Methods
    -------
    get_fpath(key: str) -> Path
        Get the file path from the key.

    return_df(key: str) -> pd.DataFrame
        Return the dataframe from the file path.
    """

    # Storing in the class instance.
    data_dict: ClassVar[dict[str, Path]] = {}

    def __repr__(self) -> str:
        rprint(self.data_dict)
        # Return an empty string.
        return ""

    def __init__(self) -> None:
        self.bdir_base = self._return_project_base_directory()

    def _return_project_base_directory(self) -> Path:
        """Return the project base directory."""
        return Path(os.getcwd())

    def get_fpath(self, key: str) -> Path:
        """Get the file path from the key."""
        if key not in self.data_dict:
            raise KeyError(f"Key {key} not found in the {list(self.data_dict.keys())}.")

        return self.data_dict.get(key, Path())

    def return_df(
        self,
        key: str,
        return_empty_df_if_missing: bool = False,
        columns_for_empty_df: list = [],
    ) -> pd.DataFrame:
        """Return the dataframe from the file path."""
        fpath_to_read = self.get_fpath(key)

        if not fpath_to_read.exists():
            if return_empty_df_if_missing:
                return pd.DataFrame(columns=columns_for_empty_df)
            raise FileNotFoundError(f"File not found: {fpath_to_read}")

        if fpath_to_read.suffix == ".parquet":
            return ParquetIO.read_parquet(fpath_to_read)
        raise ValueError(f"File type not supported: {fpath_to_read.suffix}")
