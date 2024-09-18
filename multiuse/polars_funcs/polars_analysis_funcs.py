import csv
import gzip
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import chardet
import numpy as np
import polars as pl


def get_csv_metadata(file_path: Path) -> Dict[str, Any]:
    """
    Get metadata about the CSV file without reading it entirely into memory.

    Args:
    file_path (str): Path to the CSV file

    Returns:
    dict: Metadata about the CSV file
    """
    scan = pl.scan_csv(file_path)
    schema = scan.schema
    column_names = scan.columns
    row_count = scan.select(pl.len()).collect()[0, 0]

    return {
        "file_path": file_path,
        "column_count": len(column_names),
        "column_names": column_names,
        "row_count": row_count,
        "schema": {name: str(dtype) for name, dtype in schema.items()},
    }


def is_numeric_dtype(dtype: pl.DataType) -> bool:
    return dtype in [
        pl.Int8,
        pl.Int16,
        pl.Int32,
        pl.Int64,
        pl.UInt8,
        pl.UInt16,
        pl.UInt32,
        pl.UInt64,
        pl.Float32,
        pl.Float64,
    ]


def get_column_stats(file_path: Path, column_name: str) -> Dict[str, Any]:
    """
    Get basic statistics for a specific column in the CSV file.

    Args:
    file_path (str): Path to the CSV file
    column_name (str): Name of the column to analyze

    Returns:
    dict: Statistics for the specified column
    """
    scan = pl.scan_csv(file_path)

    # First, determine the data type of the column
    dtype = scan.select(pl.col(column_name)).collect().dtypes[0]

    # Define statistics based on data type
    if is_numeric_dtype(dtype):
        stats = (
            scan.select(
                [
                    pl.col(column_name).min().alias("min"),
                    pl.col(column_name).max().alias("max"),
                    pl.col(column_name).mean().alias("mean"),
                    pl.col(column_name).std().alias("std"),
                    pl.col(column_name).n_unique().alias("unique_count"),
                ]
            )
            .collect()
            .to_dict(as_series=False)
        )
    else:
        stats = (
            scan.select(
                [
                    pl.col(column_name).min().alias("min"),
                    pl.col(column_name).max().alias("max"),
                    pl.col(column_name).n_unique().alias("unique_count"),
                ]
            )
            .collect()
            .to_dict(as_series=False)
        )

    stats["dtype"] = str(dtype)
    return {k: v[0] if isinstance(v, list) else v for k, v in stats.items()}


def query_csv(file_path: Path, query_func: Callable) -> pl.DataFrame:
    """
    Apply a custom query function to the CSV file.

    Args:
    file_path (str): Path to the CSV file
    query_func (callable): A function that takes a LazyFrame and returns a LazyFrame

    Returns:
    pl.DataFrame: Result of the query
    """
    scan = pl.scan_csv(file_path)
    return query_func(scan).collect()


def query_column_by_search_term(
    file_path: Path, column_name: str, search_term: str
) -> pl.DataFrame:
    """
    Query the CSV file to find rows where the colum_name contains search_term.

    Args:
    file_path (str): Path to the CSV file

    Returns:
    pl.DataFrame: Filtered rows containing search_term in the column_name.
    """
    # Specify dtypes for problematic columns
    dtypes = {column_name: pl.Utf8}

    scan = pl.scan_csv(
        file_path,
        dtypes=dtypes,  # Specify dtypes for problematic columns
        encoding="utf8-lossy",  # Use lossy UTF-8 encoding)
    )

    return scan.filter(pl.col(column_name).str.contains(search_term)).collect()


def is_gzipped(file_path: Path) -> bool:
    """Check if a file is gzipped."""
    with file_path.open("rb") as f:
        return f.read(2) == b"\x1f\x8b"


def detect_encoding(file_path: Path, sample_size: int = 1000000) -> str:
    """Detect the encoding of a file, whether gzipped or not."""
    if is_gzipped(file_path):
        with gzip.open(file_path, "rb") as f:
            raw = f.read(sample_size)
    else:
        with file_path.open("rb") as f:
            raw = f.read(sample_size)
    return chardet.detect(raw)["encoding"]


def preprocess_large_csv(
    input_path: Path, output_path: Path, chunk_size: int = 1000000
) -> None:
    """
    Preprocess a large CSV file efficiently, handling both .csv and .csv.gz files.

    Args:
    input_path (Path): Path to the input CSV file (can be .csv or .csv.gz)
    output_path (Path): Path to save the preprocessed CSV file
    chunk_size (int): Number of lines to process at a time
    """
    # Detect if the file is gzipped
    is_gz = is_gzipped(input_path)

    # Detect the encoding of the input file
    encoding = detect_encoding(input_path)
    print(f"Detected encoding: {encoding}")

    # Open the input file based on whether it's gzipped or not
    if is_gz:
        infile = gzip.open(input_path, "rt", encoding=encoding, errors="replace")
    else:
        infile = input_path.open("r", encoding=encoding, errors="replace")

    with infile, output_path.open("w", newline="", encoding="utf-8") as outfile:
        # Create CSV reader and writer
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        # Write the header
        header = next(reader)
        writer.writerow(header)

        # Process the file in chunks
        chunk = []
        for row in reader:
            chunk.append(row)
            if len(chunk) >= chunk_size:
                writer.writerows(chunk)
                chunk = []

        # Write any remaining rows
        if chunk:
            writer.writerows(chunk)


def scan_csv_path(
    file_path: Path, columns_dtypes: Dict[str, pl.DataType]
) -> pl.DataFrame:
    """
    Scan a CSV file and return a lazy DataFrame.

    Args:
    file_path (Path): Path to the CSV file

    Returns:
    pl.DataFrame: Lazy DataFrame representing the CSV file
    """

    return pl.scan_csv(
        file_path,
        dtypes=columns_dtypes,  # Specify dtypes for problematic columns
        encoding="utf8-lossy",  # Use lossy UTF-8 encoding
    )


def query_columns_with_exact_and_fuzzy(
    file_path: Path,
    exact_match: Optional[Dict[str, List[Union[str, int]]]] = None,
    fuzzy_match: Optional[Dict[str, List[str]]] = None,
) -> pl.DataFrame:
    """
    Query the CSV file to find rows that match both exact and fuzzy criteria, optimized for large files.

    Args:
    file_path (Path): Path to the CSV file
    exact_match (Dict[str, List[Union[str, int]]]): Dictionary of columns and values for exact matching
    fuzzy_match (Dict[str, List[str]]): Dictionary of columns and values for fuzzy matching

    Returns:
    pl.DataFrame: Filtered DataFrame containing matching rows

    Examples:
    >>> exact_match = {"COLLATERAL": ["SNAPON"]}
    >>> fuzzy_match = {"SEC_PARTY": ["SNAPON", "SNAP-ON", "SNAP ON"]}

    >>> query_columns_with_exact_and_fuzzy(file_path, exact_match, fuzzy_match)
    """
    if not exact_match and not fuzzy_match:
        raise ValueError("At least one of exact_match or fuzzy_match must be provided")

    if not isinstance(exact_match, dict):
        raise ValueError("exact_match must be a dictionary")

    # Determine columns to read
    columns_to_read = list(
        set(list(exact_match.keys()) + list(fuzzy_match.keys()) if fuzzy_match else [])
    )

    # Specify dtypes for columns
    dtypes = {col: pl.Utf8 for col in columns_to_read}

    # Create scan object
    scan = pl.scan_csv(
        file_path, dtypes=dtypes, encoding="utf8-lossy", ignore_errors=True
    )

    # Build filter expression
    filter_expr = pl.lit(True)

    # Apply exact matching
    if exact_match:
        for column, values in exact_match.items():
            filter_expr &= pl.col(column).is_in([str(v) for v in values])

    # Apply fuzzy matching
    if fuzzy_match:
        for column, patterns in fuzzy_match.items():
            combined_pattern = "|".join(patterns)
            filter_expr &= pl.col(column).str.contains(combined_pattern, literal=False)

    # Apply filter and collect results
    result = scan.filter(filter_expr).collect()

    # Remove duplicates
    result = result.unique()
    print(f"Total rows found (after deduplication): {len(result)}")

    return result


def query_column_by_search_terms(
    file_path: Path, column_name: str, search_terms: list[str]
) -> pl.DataFrame:
    """
    Query the CSV file to find rows where the specified column contains any of the search terms.

    Args:
    file_path (Path): Path to the CSV file
    column_name (str): Name of the column to search
    search_terms (list[str]): List of terms or regex patterns to search for in the column

    Returns:
    pl.DataFrame: Filtered rows containing any of the search terms in the specified column
    """
    # Specify dtypes for problematic columns
    dtypes = {column_name: pl.Utf8}

    # Combine search terms into a single regex pattern
    combined_pattern = "|".join(search_terms)

    scan = pl.scan_csv(
        file_path,
        dtypes=dtypes,  # Specify dtypes for problematic columns
        encoding="utf8-lossy",  # Use lossy UTF-8 encoding
    )

    return scan.filter(
        pl.col(column_name).str.contains(combined_pattern, literal=False)
    ).collect()


def query_columns_by_search_terms(
    file_path: Path,
    columns: list[str],
    search_terms: list[str],
    case_sensitive: bool = False,
) -> pl.DataFrame:
    """
    Query the CSV file to find rows where the specified columns contain any of the search terms.

    Args:
    file_path (Path): Path to the CSV file
    columns (list[str]): Names of the columns to search
    search_terms (list[str]): List of terms or regex patterns to search for in the columns
    case_sensitive (bool): Whether the search should be case-sensitive (default: False)

    Returns:
    dict[str, pl.DataFrame]: Dictionary with column names as keys and filtered DataFrames as values
    """
    # Specify dtypes for columns
    dtypes = {col: pl.Utf8 for col in columns}

    # Combine search terms into a single regex pattern
    combined_pattern = "|".join(search_terms)

    scan = pl.scan_csv(
        file_path, dtypes=dtypes, encoding="utf8-lossy", ignore_errors=True
    )

    # Build the filter expression for all specified columns
    filter_expr = pl.lit(False)
    for column in columns:
        if case_sensitive:
            filter_expr |= pl.col(column).str.contains(combined_pattern, literal=False)
        else:
            filter_expr |= (
                pl.col(column)
                .str.to_lowercase()
                .str.contains(combined_pattern.lower(), literal=False)
            )

    # Apply the filter and collect results
    result = scan.filter(filter_expr).collect()

    # Remove duplicates
    result = result.unique()

    print(f"Total rows found (after deduplication): {len(result)}")

    return result


def query_columns_by_coordinate_pairs(
    file_path: Path,
    columns: list[str],
    coordinate_pairs: list[tuple[float, float]],
    tolerance: float = 0.0001,
) -> pl.DataFrame:
    """
    Query the CSV file to find rows where the specified columns match the given coordinate pairs within a tolerance.

    Args:
    file_path (Path): Path to the CSV file
    columns (list[str]): Names of the two columns to search (latitude and longitude)
    coordinate_pairs (list[tuple[float, float]]): List of (latitude, longitude) pairs to search for
    tolerance (float): The maximum allowed difference between coordinates (default: 0.0001)

    Returns:
    pl.DataFrame: Filtered DataFrame containing matching rows
    """
    if len(columns) != 2:
        raise ValueError(
            "Exactly two column names must be provided: [latitude, longitude]"
        )

    # Specify dtypes for columns
    dtypes = {col: pl.Float64 for col in columns}

    scan = pl.scan_csv(
        file_path, dtypes=dtypes, encoding="utf8-lossy", ignore_errors=True
    )

    # Build the filter expression for the coordinate pairs
    filter_expr = pl.lit(False)
    for lat, lon in coordinate_pairs:
        lat_condition = (pl.col(columns[0]) >= lat - tolerance) & (
            pl.col(columns[0]) <= lat + tolerance
        )
        lon_condition = (pl.col(columns[1]) >= lon - tolerance) & (
            pl.col(columns[1]) <= lon + tolerance
        )
        filter_expr |= lat_condition & lon_condition

    # Apply the filter and collect results
    result = scan.filter(filter_expr).collect()

    # Remove duplicates
    result = result.unique()
    print(f"Total rows found (after deduplication): {len(result)}")

    return result


def write_results_to_csv(results: pl.DataFrame, output_path: Path) -> None:
    """
    Write the query results to a CSV file.

    Args:
    results (pl.DataFrame): The DataFrame containing the query results
    output_path (Path): The path where the CSV file should be saved
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results.write_csv(output_path)
    print(f"Results written to {output_path}: {len(results)} rows")


def get_value_counts_lazy(csv_path: Path, column_name: str) -> pl.DataFrame:
    """
    Get the value counts for a specific column in a lazy DataFrame.

    Args:
    df (pl.DataFrame): The input DataFrame
    column_name (str): The name of the column to compute value counts for

    Returns:
    pl.DataFrame: A DataFrame containing the value counts for the specified column
    """
    return (
        scan_csv_path(csv_path, {column_name: str})
        .select(pl.col(column_name).value_counts(sort=True))
        .collect()
    )


def get_multi_column_value_counts_lazy(
    csv_path: Path, column_names: list[str]
) -> pl.DataFrame:
    """
    Get the combined value counts for multiple specified columns in a lazy DataFrame.

    Args:
    csv_path (Path): The path to the CSV file
    column_names (list[str]): A list of column names to compute combined value counts for

    Returns:
    pl.DataFrame: A DataFrame containing the combined value counts for the specified columns
    """
    # Create a dictionary to store the schema for scan_csv
    schema = {col: str for col in column_names}

    # Scan the CSV file
    df = scan_csv_path(csv_path, schema)

    # Compute combined value counts for the specified columns
    return (
        df.group_by(column_names)
        .agg(pl.count().alias("count"))
        .sort("count", descending=True)
        .collect()
    )


def format_vc_for_two_columns(value_counts: pl.DataFrame) -> pl.DataFrame:
    """
    Format the value counts DataFrame for two columns.

    Args:
    value_counts (pl.DataFrame): DataFrame containing the value counts

    Returns:
    pl.DataFrame: Formatted Polars DataFrame with two columns
    """
    df_value_counts = value_counts.to_pandas()

    df_value_counts["party"] = df_value_counts["SEC_PARTY"].apply(
        lambda x: x["SEC_PARTY"]
    )

    df_value_counts["count"] = df_value_counts["SEC_PARTY"].apply(lambda x: x["count"])

    df_value_counts = df_value_counts.drop(columns=["SEC_PARTY"])

    # Now convert back to Polars DataFrame
    return pl.DataFrame(df_value_counts)


def convert_column_to_json(df: pl.DataFrame, column_name: str) -> pl.DataFrame:
    """
    Convert a specified column in a Polars DataFrame to JSON strings.

    Args:
    df (pl.DataFrame): Input Polars DataFrame
    column_name (str): Name of the column to convert to JSON strings

    Returns:
    pl.DataFrame: Modified Polars DataFrame with the specified column converted to JSON strings
    """

    def convert_to_json(x: Any) -> str:
        if isinstance(x, np.ndarray):
            x = x.tolist()
        return json.dumps(x)

    # Convert Polars DataFrame to Pandas
    df_pandas = df.to_pandas()

    # Apply JSON conversion to the specified column
    df_pandas[column_name] = df_pandas[column_name].apply(convert_to_json)

    # Convert back to Polars DataFrame
    return pl.DataFrame(df_pandas)


def add_associated_names(
    csv_path: Path,
    count_df: pl.DataFrame,
    join_columns: list[str],
    party_column: str = "SEC_PARTY",
) -> pl.DataFrame:
    """
    Join the count DataFrame with the full DataFrame from CSV and add associated names as JSON.

    Args:
    csv_path (Path): Path to the CSV file containing the full data
    count_df (pl.DataFrame): The DataFrame with value counts
    join_columns (list[str]): The columns to join on (e.g., ["column1", "column2"])
    party_column (str): The name of the column containing party names (default: "SEC_PARTY")

    Returns:
    pl.DataFrame: A DataFrame with the original counts and a new column of associated names as JSON
    """
    # Create a schema for the CSV scan
    schema = {col: str for col in join_columns}
    schema[party_column] = str

    # Scan the CSV file
    full_df_lazy = scan_csv_path(csv_path, schema)

    # Group the full DataFrame by join columns and aggregate party names into a list
    associated_names = full_df_lazy.group_by(join_columns).agg(
        pl.col(party_column).alias("temp_associated_names")
    )

    # Convert count_df to lazy
    count_df_lazy = count_df.lazy()

    # Join the count DataFrame with the associated names
    result = count_df_lazy.join(associated_names, on=join_columns, how="left")

    # Add the associated_names column
    result = result.with_columns(
        [pl.col("temp_associated_names").fill_null([]).alias("associated_names_filled")]
    )

    # Make the list unique
    result = result.with_columns(
        [pl.col("associated_names_filled").list.unique().alias("associated_names")]
    )

    # Drop the temporary columns
    result = result.drop(["temp_associated_names", "associated_names_filled"])

    return result.collect()
