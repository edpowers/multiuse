import contextlib
import re
from functools import partial
from itertools import batched
from pathlib import Path
from typing import cast

import duckdb
import polars as pl
from joblib import Parallel, delayed

EXCLUDED_SEARCH_TERMS = [
    "MARINE",
    "MERCURY",
    "SKIFF",
    "OIL",
    "TRAILER",
    "BOAT",
    "HULL",
    "BIMBOX STRYKER",
    "STRYKER LOGISTICS",
    "TREADMILL",
    "TRUE FITNESS",
    "PARCEL",
    "ALUMAWELD",
    "YAMAHA",
    "KUBOTA",
    "SUZUKI",
    "FIBERGLASS",
    "HONDA",
    "CARHAULER",
    "MOTOR",
    "Aqua Finance",
    "Cg Automation And Fixture",
    "CHOPPER",
    "SNOWPLOW",
    "TAHOE",
    "DEFENDER",
    "STRYKER-MUNLEY",
    "VIN#",
    "VIN/",
    "SENIOR HOUSING",
    "STRYKER STREET",
    "Farm Bureau Bank FSB",
    "KAWASAKI",
    "SUZUKI",
    "HONDA",
    "ARTICAT",
    "KUBOTA",
    "YAMAHA",
    "POLARIS",
    "Tobacco",
    "MOWER",
    "WHEELER",
    "BOBCATLOADER",
    "LEEBOY",
    "TRAILER",
    "FORKLIFT",
    "VEHICLE",
    "PNEUMATIC",
    "NOMAD DONUTS",
    "Envista Credit Union",
    "Envista CU",
    "NOMAD GROUP",
    "Envista Federal",
    "REAL ESTATE",
    "Envista Federal Credit Union",
    "CAMPER",
    " VIN ",
    " VIN#",
    " V1N ",
    "SURVEY PRO",
    "FIBERGLASS",
]


def collect_df(lf: pl.LazyFrame) -> pl.DataFrame:
    return lf.collect()  # type: ignore[return-value]


# Value counts for each search term in the results
def get_search_term_counts(
    df: pl.DataFrame,
    search_terms: list[str],
    text_columns: list[str] | None = None,
    use_n_rows: int = 5_000,
) -> pl.DataFrame:
    """
    Count how many times each search term appears in the specified text columns.

    Args:
        df: The dataframe to search through
        search_terms: List of search terms to count
        text_columns: List of text columns to search in

    Returns:
        DataFrame with search_term and count columns
    """
    counts = []

    if not text_columns:
        text_columns = ["COLLATERAL", "SEC_PARTY"]

    if use_n_rows:
        df = df.head(use_n_rows)

    for term in search_terms:
        # Create a boolean mask for each text column that contains the search term
        mask = pl.lit(False)
        for col in text_columns:
            if col in df.columns:
                mask = mask | pl.col(col).str.to_lowercase().str.contains(
                    term.lower(),
                    literal=True,
                )

        # Count rows where the term appears in any of the text columns
        count = df.filter(mask).height
        counts.append({"search_term": term, "count": count})

    return (
        pl.DataFrame(counts)
        .filter(pl.col("count").gt(0))
        .sort("count", descending=True)
    )


def format_search_string(
    search_string: str | list[str],
    convert_to_lowercase: bool = False,
    use_regex: bool = True,
) -> str:
    """
    Format search string(s) into a regex pattern with word boundaries.

    Args:
        search_string: Single string or list of strings to search for
        convert_to_lowercase: Whether to convert all strings to lowercase
        use_regex: If False, escape all regex special characters

    Returns:
        Formatted regex pattern string
    """
    # If already a formatted regex pattern, return as-is
    if isinstance(search_string, str) and (
        "\\" in search_string or any(c in search_string for c in ".*+?{}[]()^$|")
    ):
        return search_string

    # Normalize to list for consistent processing
    if isinstance(search_string, str):
        search_list = [search_string]
    else:
        search_list = list(search_string)  # Create copy to avoid modifying original

    # Apply lowercase conversion if requested
    if convert_to_lowercase:
        search_list = [name.lower() for name in search_list]

    # Process each search term
    processed_terms = []
    for name in search_list:
        # Skip empty strings
        if not name:
            continue

        if use_regex and any(c in name for c in ".*+?{}[]()^$|\\"):
            # Keep regex patterns as-is (already contains regex syntax)
            processed_terms.append(name)
        else:
            # Clean special characters and escape for regex
            cleaned_name = (
                name.replace("(", "")
                .replace(")", "")
                .replace("[", "")
                .replace("]", "")
                .replace("*", "")
            )
            # Escape remaining special regex characters
            escaped_name = re.escape(cleaned_name)
            processed_terms.append(escaped_name)

    # Handle empty result
    if not processed_terms:
        ve_string = "No valid search terms provided"
        raise ValueError(ve_string)

    # Build final regex with word boundaries (using raw string for clarity)
    # Use non-capturing groups for efficiency

    return rf"(?:^|\b|\s)(?:{'|'.join(processed_terms)})(?:\b|\s|$)"


def write_results_to_csv(
    results: pl.DataFrame,
    output_path: Path,
    print_results: bool = True,
    write_if_empty: bool = False,
) -> None:
    """
    Write the query results to a CSV file.

    Args:
    results (pl.DataFrame): The DataFrame containing the query results
    output_path (Path): The path where the CSV file should be saved
    print_results (bool, default = True): Whether to print the results
    write_if_empty (bool, default = False): Whether to write the results if they are empty
    """
    if not write_if_empty and len(results) == 0:
        return

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)

    results.write_csv(output_path)

    if print_results:
        print(f"Results written to {output_path}: {len(results)} rows")


# OPTIMIZATION: For substring matching with many terms
# Split into chunks and use separate conditions
def create_substring_conditions(
    col_name: str,
    terms: list[str],
    chunk_size: int = 10,
    use_regex: bool = True,
    case_sensitive: bool = False,
) -> pl.Expr:
    """
    Creates optimized substring matching conditions.
    For large term lists, splits into chunks to avoid regex performance issues.
    """
    # Lowercase column.
    lowercase_column = pl.col(col_name).str.to_lowercase()

    if len(terms) <= chunk_size:
        # For small lists, use single regex (still performant)
        pattern = format_search_string(terms, convert_to_lowercase=not case_sensitive)
        return lowercase_column.str.contains(pattern)
    # For large lists, split into chunks
    chunk_conditions = []

    for i in range(0, len(terms), chunk_size):
        chunk = terms[i : i + chunk_size]

        pattern = format_search_string(chunk, convert_to_lowercase=not case_sensitive)
        chunk_conditions.append(lowercase_column.str.contains(pattern, literal=False))

    # Combine all chunks with OR logic
    return pl.any_horizontal(chunk_conditions)


def find_rows_with_phrase_df(
    df: pl.DataFrame | pl.LazyFrame,
    phrase: list[str],
    columns: list[str] | None = None,
    exclude: bool = False,
    case_sensitive: bool = False,
    debug: bool = False,
    return_original_if_all_excluded: bool = False,
) -> pl.DataFrame | pl.LazyFrame:
    if exclude and not phrase:
        if debug:
            print(
                "Returning original without exclusion since no exclude terms provided.",
            )
        return df

    if not isinstance(df, pl.LazyFrame):
        df = df.lazy()

    # Store initial count for comparison
    initial_count = collect_df(df).item()
    # Validate columns exist
    if not columns:
        columns = ["COLLATERAL"]

    if exclude:
        processed_phrase = format_search_string(
            phrase,
            convert_to_lowercase=not case_sensitive,
        )
        result = df.clone()
        for column in columns:
            result = result.filter(
                ~pl.col(column).str.to_lowercase().str.contains(processed_phrase),
            )
    else:
        result = df.filter(
            pl.any_horizontal(
                [
                    create_substring_conditions(
                        column,
                        phrase,
                        chunk_size=10,
                        case_sensitive=case_sensitive,
                    )
                    for column in columns
                ],
            ),
        )

    # Get final count and calculate excluded rows
    result_final = collect_df(result)
    excluded_count = initial_count - result_final.item()

    if exclude and excluded_count > 0 and return_original_if_all_excluded:
        print(
            f"Excluded {excluded_count:,} rows ({(excluded_count / initial_count) * 100:.2f}% of total)",
        )

        # If everything was excluded.
        if excluded_count == initial_count:
            print("No rows were found after exclusion. Returning original.")
            return df

    return result_final


def align_schema(
    df: pl.DataFrame,
    target_schema: dict,
    fill_missing_columns: bool = True,
) -> pl.DataFrame:
    """
    Align dataframe schema with target schema, including column order.

    Args:
        df: DataFrame to modify
        target_schema: Target schema to match

    Returns:
        DataFrame with aligned schema and columns ordered according to target_schema
    """
    # First handle type conversions
    for col_name, dtype in target_schema.items():
        if col_name in df.columns:
            # Handle specific type conversions
            if dtype == pl.String and df[col_name].dtype == pl.Int64:
                df = df.with_columns(pl.col(col_name).cast(pl.String))
            elif dtype == pl.Int64 and df[col_name].dtype == pl.String:
                df = df.with_columns(
                    pl.col(col_name)
                    .str.replace_all(r"^\s*$", "0")
                    .cast(pl.Int64, strict=False),
                )
            elif dtype in (pl.Float64, pl.Float32) and df[col_name].dtype == pl.String:
                df = df.with_columns(
                    pl.col(col_name)
                    .str.replace_all(r"^\s*$", "0.0")
                    .cast(dtype, strict=False),
                )
            else:
                df = df.with_columns(pl.col(col_name).cast(dtype))

        elif fill_missing_columns:
            df = df.with_columns(
                pl.Series(name=col_name, values=[None] * len(df), dtype=dtype),
            )

    # Reorder columns to match target schema
    # Only include columns that exist in the DataFrame
    # Reorder columns to match target schema
    # Only include columns that exist in the DataFrame
    target_cols = [col for col in target_schema if col in df.columns]

    return df.select(target_cols)


def find_rows_with_phrase_from_fpath(
    fpath: Path,
    search_terms: list[str],
    columns_to_search: list[str],
    lazy: bool = True,
    read_all_columns: bool = False,
    additional_columns: list[str] | None = None,
    use_regex: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    """
    Find rows in a parquet file that contain any of the specified search terms.

    Args:
        fpath: Path to the parquet file
        search_terms: List of terms to search for
        columns_to_search: Columns to search in
        lazy: If True, returns a LazyFrame; if False, returns a DataFrame

    Returns:
        A LazyFrame or DataFrame containing the matching rows
    """
    if read_all_columns:
        scan_df = pl.scan_parquet(fpath, low_memory=True)
    else:
        scan_df = pl.scan_parquet(fpath, low_memory=True).select(
            list(
                set(
                    columns_to_search
                    + ["FILE_DATE", "ROW_INDEX"]
                    + (additional_columns or []),
                ),
            ),
        )

    result = find_rows_with_phrase_df(
        df=scan_df,
        columns=columns_to_search,
        phrase=search_terms,
    )

    del scan_df

    # Only collect if explicitly requested
    if not lazy and isinstance(result, pl.LazyFrame):
        return collect_df(result)

    return result


def find_rows_with_phrase_duckdb(
    fpath: Path | str,
    search_terms: str | list[str],
    columns_to_search: list[str],
    exclude: bool = False,
    case_sensitive: bool = False,
    use_regex: bool = True,
    read_all_columns: bool = False,
    additional_columns: list[str] | None = None,
    word_boundary: bool = True,
    debug: bool = False,
) -> pl.DataFrame:
    """DuckDB text search with combined regex pattern."""

    if exclude and not search_terms:
        conn = duckdb.connect()
        return conn.execute("SELECT * FROM read_parquet(?)", [str(fpath)]).pl()

    # Normalize
    if isinstance(search_terms, str):
        search_terms = [search_terms]

    if not columns_to_search:
        columns_to_search = ["COLLATERAL"]

    # Column selection
    if read_all_columns:
        select_cols = "*"
    else:
        cols = list(
            set(
                columns_to_search
                + ["FILE_DATE", "ROW_INDEX"]
                + (additional_columns or []),
            ),
        )
        select_cols = ", ".join(cols)

    conn = duckdb.connect()

    # Build single combined pattern
    if use_regex:
        bounded_terms = search_terms  # Use as-is for regex
    elif word_boundary:
        bounded_terms = [rf"\b{re.escape(term)}\b" for term in search_terms]
    else:
        bounded_terms = [re.escape(term) for term in search_terms]

    combined_pattern = ("(?i)" if not case_sensitive else "") + "|".join(bounded_terms)

    # Single condition per column with combined pattern
    col_conditions = [f"regexp_matches({col}, ?)" for col in columns_to_search]

    # Join columns with OR (match in ANY column)
    where_clause = " OR ".join(col_conditions)

    if exclude:
        where_clause = f"NOT ({where_clause})"

    params = [str(fpath)] + [combined_pattern] * len(columns_to_search)

    query = f"""
    SELECT {select_cols}
    FROM read_parquet(?)
    WHERE {where_clause}
    """

    if debug:
        print(f"Pattern: {combined_pattern}")
        print(f"Query: {query}")

    result = conn.execute(query, params)
    return result.pl()


def find_rows_by_indices_duckdb(
    fpath: Path | str,
    row_indices: list[int],
    read_all_columns: bool = False,
    additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """Filter parquet by exact ROW_INDEX matches (cast from string to int64)."""

    if not row_indices:
        return pl.DataFrame()

    conn = duckdb.connect()

    if read_all_columns:
        select_cols = "*"
    else:
        cols = list(set(["ROW_INDEX", "FILE_DATE"] + (additional_columns or [])))
        select_cols = ", ".join(cols)

    # DuckDB IN clause with cast
    indices_str = ",".join(map(str, row_indices))

    query = f"""
    SELECT {select_cols}
    FROM read_parquet(?)
    WHERE CAST(ROW_INDEX AS BIGINT) IN ({indices_str})
    """

    return conn.execute(query, [str(fpath)]).pl()


def search_partitioned_parquet(
    base_path: str | Path,
    search_terms: list[str] | None = None,
    columns_to_search: list[str] | None = None,
    partition_by: str = "FILE_YEAR",
    partition_values: list[int] | None = None,
    exclude_terms: list[str] | None = None,
    additional_columns: list[str] | None = None,
    read_all_columns: bool = False,
    n_jobs: int = -4,
    use_duckdb: bool = True,
    target_schema: dict[str, pl.DataType] | None = None,
    search_by_indices: bool = False,
    use_regex: bool = True,
    word_boundary: bool = True,
    debug: bool = False,
    batch_size: int | None = None,  # NEW
    dedup_columns: list[str] | None = None,  # NEW - defaults to ["ROW_INDEX"]
    display_vc_counts: bool = False,
    batch_size_constant: int = 50,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Efficiently search through partitioned parquet files for specific terms.

    New params:
        batch_size: If provided, processes search_terms in batches of this size
        dedup_columns: Columns to use for deduplication (default: ["ROW_INDEX"])
    """
    dedup_columns = dedup_columns or ["ROW_INDEX"]

    # Default to 50 as batch size if search terms > 50.
    if search_terms and len(search_terms) > batch_size_constant:
        batch_size = batch_size_constant

    # If batching requested, process in batches
    if batch_size and search_terms and len(search_terms) > batch_size:
        all_results = []
        all_excluded = []

        for batch in batched(search_terms, batch_size):
            results, excluded = _search_single_batch(
                base_path=base_path,
                search_terms=list(batch),
                columns_to_search=columns_to_search,
                partition_by=partition_by,
                partition_values=partition_values,
                exclude_terms=exclude_terms,
                additional_columns=additional_columns,
                read_all_columns=read_all_columns,
                n_jobs=n_jobs,
                use_duckdb=use_duckdb,
                search_by_indices=search_by_indices,
                target_schema=target_schema,
                use_regex=use_regex,
                word_boundary=word_boundary,
                debug=debug,
            )

            if display_vc_counts:
                vc_counts = get_search_term_counts(results, batch, use_n_rows=2_500)
                if not vc_counts.is_empty():
                    print(vc_counts)

            all_results.append(results)
            all_excluded.append(excluded)

        # Deduplicate and combine
        combined_results = (
            (
                pl.concat(all_results, how="vertical_relaxed").unique(
                    subset=dedup_columns,
                )
            )
            if all_results
            else pl.DataFrame()
        )

        excluded_rows = (
            (
                pl.concat(all_excluded, how="vertical_relaxed").unique(
                    subset=dedup_columns,
                )
            )
            if all_excluded
            else pl.DataFrame()
        )

        return combined_results, excluded_rows

    # No batching - use original logic
    return _search_single_batch(
        base_path=base_path,
        search_terms=search_terms,
        columns_to_search=columns_to_search,
        partition_by=partition_by,
        partition_values=partition_values,
        exclude_terms=exclude_terms,
        additional_columns=additional_columns,
        read_all_columns=read_all_columns,
        n_jobs=n_jobs,
        use_duckdb=use_duckdb,
        search_by_indices=search_by_indices,
        target_schema=target_schema,
        use_regex=use_regex,
        word_boundary=word_boundary,
        debug=debug,
    )


def _search_single_batch(
    base_path: str | Path,
    search_terms: list[str] | None,
    columns_to_search: list[str] | None,
    partition_by: str,
    partition_values: list[int] | None,
    exclude_terms: list[str] | None,
    additional_columns: list[str] | None,
    read_all_columns: bool,
    n_jobs: int,
    use_duckdb: bool,
    search_by_indices: bool,
    target_schema: dict[str, pl.DataType] | None,
    use_regex: bool,
    word_boundary: bool,
    debug: bool,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Original search logic - processes a single batch of terms."""
    base_path = Path(base_path)
    if not columns_to_search:
        ve_string = "columns_to_search cannot be none"
        raise ValueError(ve_string)
    if not search_terms:
        ve_string = "search terms cannot be none"
        raise ValueError(ve_string)

    columns_to_load = list(set(columns_to_search + (additional_columns or [])))

    # Discover partitions
    if partition_values is None:
        partition_values = []
        for path in base_path.glob(f"{partition_by}=*"):
            if path.is_dir():
                try:
                    value = path.name.split("=")[1]
                    with contextlib.suppress(ValueError):
                        value = int(value)
                    partition_values.append(value)
                except IndexError:
                    continue

    # Build paths
    paths_to_read = []
    for value in partition_values:
        partition_dir = base_path / f"{partition_by}={value}"
        if partition_dir.exists():
            parquet_files = list(partition_dir.rglob("*.parquet"))
            paths_to_read.extend([str(p) for p in parquet_files])

    if not paths_to_read:
        ve_string = f"No parquet files found for partition values: {partition_values}"
        raise ValueError(ve_string)

    # Process files in parallel
    if use_duckdb:
        if search_by_indices:
            process_fn = partial(
                find_rows_by_indices_duckdb,
                row_indices=cast("list[int]", search_terms),
                read_all_columns=True,
            )
        else:
            process_fn = partial(
                find_rows_with_phrase_duckdb,
                search_terms=search_terms,
                columns_to_search=columns_to_search,
                case_sensitive=False,
                additional_columns=additional_columns,
                debug=debug,
                read_all_columns=read_all_columns,
                word_boundary=word_boundary,
                use_regex=use_regex,
            )
    else:
        process_fn = partial(
            find_rows_with_phrase_from_fpath,
            search_terms=search_terms,
            columns_to_search=columns_to_search,
            lazy=False,
            additional_columns=additional_columns,
            read_all_columns=read_all_columns,
            use_regex=use_regex,
        )

    results_list_raw = Parallel(n_jobs=n_jobs)(
        delayed(process_fn)(path) for path in paths_to_read
    )

    # Align schemas
    if read_all_columns:
        all_columns = {col for df in results_list_raw for col in df.columns}
        target_schema: dict[str, type[pl.DataType]] = dict.fromkeys(
            all_columns,
            pl.Utf8,
        )
    else:
        columns_to_load = list(set(columns_to_load + (additional_columns or [])))
        if target_schema is None:
            target_schema: dict[str, type[pl.DataType]] = dict.fromkeys(
                columns_to_load,
                pl.Utf8,
            )

    results_list = [
        align_schema(df, target_schema=target_schema) for df in results_list_raw
    ]

    # Combine and apply exclusions
    if results_list:
        combined_results = pl.concat(results_list, how="vertical_relaxed")

        if exclude_terms:
            excluded_rows = find_rows_with_phrase_df(
                df=combined_results,
                columns=columns_to_search,
                phrase=exclude_terms,
                exclude=False,
            )
            combined_results = find_rows_with_phrase_df(
                df=combined_results,
                columns=columns_to_search,
                phrase=exclude_terms,
                exclude=True,
            )
        else:
            excluded_rows = combined_results.head(0)
    else:
        combined_results = pl.DataFrame()
        excluded_rows = pl.DataFrame()

    if isinstance(combined_results, pl.LazyFrame):
        combined_results = combined_results.collect()
    if isinstance(excluded_rows, pl.LazyFrame):
        excluded_rows = excluded_rows.collect()

    return cast("pl.DataFrame", combined_results), cast("pl.DataFrame", excluded_rows)
