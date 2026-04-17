"""Basic utils for writing to parquet."""

from pathlib import Path

import polars as pl


def append_to_parquet(
    df_new: pl.DataFrame,
    path: Path,
    key_col: str | None = None,
    dedupe_cols: list[str] | None = None,
) -> pl.DataFrame:
    """Append new records to an existing parquet file, deduplicating on key_col."""
    if path.exists():
        df_existing = pl.read_parquet(path)
        # Only keep new records not already in the file
        if key_col:
            df_new = df_new.filter(~pl.col(key_col).is_in(df_existing[key_col]))

        df_combined = pl.concat([df_existing, df_new], how="diagonal_relaxed")

        if dedupe_cols:
            df_combined = df_combined.unique(dedupe_cols)
    else:
        df_combined = df_new

    df_combined.write_parquet(path, mkdir=True)

    return df_combined

