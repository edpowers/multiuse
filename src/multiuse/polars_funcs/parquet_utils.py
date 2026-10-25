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
    df_nu = df_new.lazy()

    if path.exists():
        df_existing = pl.scan_parquet(path)
        # Only keep new records not already in the file
        if key_col:
            df_nu = df_nu.join(
                df_existing.select(key_col),
                on=key_col,
                how="anti",
            )

        df_combined = pl.concat([df_existing, df_nu], how="diagonal_relaxed")

        if dedupe_cols:
            df_combined = df_combined.unique(dedupe_cols)
    else:
        df_combined = df_nu

    # Collect before writing.  The plan scans `path`, so streaming the result
    # straight back into `path` truncates the file while it is still being
    # read, which aborts in the parquet reader.
    df_result = df_combined.collect()
    path.parent.mkdir(parents=True, exist_ok=True)
    df_result.write_parquet(path)

    return df_result
