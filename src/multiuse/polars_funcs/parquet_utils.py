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
    df_nu = df_new.lazy() if not isinstance(df_new, pl.LazyFrame) else df_new

    if path.exists():
        df_existing = pl.scan_parquet(path)
        # Only keep new records not already in the file
        if key_col:
            df_nu = (
                df_nu.lazy()
                .join(df_existing.select(key_col), on=key_col, how="anti")
                .collect()
            )

        df_combined = pl.concat([df_existing, df_nu], how="diagonal_relaxed")

        if dedupe_cols:
            df_combined = df_combined.unique(dedupe_cols)
    else:
        df_combined = df_nu

    df_combined.sink_parquet(path, mkdir=True)

    df_return = df_combined.collect()
    if not isinstance(df_return, pl.DataFrame):
        te_msg = "Expected a Polars DataFrame"
        raise TypeError(te_msg)

    return df_return
