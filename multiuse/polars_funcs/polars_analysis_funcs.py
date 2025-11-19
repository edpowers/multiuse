import contextlib
import csv
import gzip
import json
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import chardet
import duckdb
import numpy as np
import polars as pl
from joblib import Parallel, delayed
from rich import print as rprint


def count_search_term_occurrences(
    df: pl.DataFrame | pl.LazyFrame,
    search_terms: list[str],
    columns_to_search: list[str],
) -> pl.DataFrame:
    """Count occurrences of each search term in specified columns.

    Args:
        df: Input DataFrame or LazyFrame
        search_terms: List of search terms to count
        columns_to_search: List of column names to search within

    Returns:
        DataFrame with search terms and their counts
    """
    # Convert to LazyFrame if DataFrame
    lazy_df = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Create expressions for counting each search term in each column
    count_exprs = []
    for term in set(search_terms):
        # Sum the counts across all specified columns for each term
        term_counts = sum([pl.col(col).str.count_matches(term).alias(f"{term}_{col}") for col in columns_to_search])
        count_exprs.append(term_counts.alias(f"count_{term}"))

    # Calculate all counts in one pass
    counts_df = lazy_df.select(count_exprs).sum().collect()

    # Reshape to long format for better visualization
    result_df = (
        counts_df.melt()
        .with_columns(
            [
                pl.col("variable").str.replace("count_", "").alias("search_term"),
                pl.col("value").alias("count"),
            ]
        )
        .select(["search_term", "count"])
        .sort("count", descending=True)
    )

    return result_df


# Value counts for each search term in the results
def get_search_term_counts(
    df: pl.DataFrame,
    search_terms: list[str],
    text_columns: list[str] = ["COLLATERAL", "SEC_PARTY"],
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

    for term in search_terms:
        # Create a boolean mask for each text column that contains the search term
        mask = pl.lit(False)
        for col in text_columns:
            if col in df.columns:
                mask = mask | pl.col(col).str.to_lowercase().str.contains(term.lower(), literal=True)

        # Count rows where the term appears in any of the text columns
        count = df.filter(mask).height
        counts.append({"search_term": term, "count": count})

    return pl.DataFrame(counts).filter(pl.col("count").gt(0)).sort("count", descending=True)


def get_csv_metadata(file_path: Path) -> dict[str, Any]:
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


def get_column_stats(file_path: Path, column_name: str) -> dict[str, Any]:
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


def query_column_by_search_term(file_path: Path, column_name: str, search_term: str) -> pl.DataFrame:
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


def preprocess_large_csv(input_path: Path, output_path: Path, chunk_size: int = 1000000) -> None:
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


def scan_csv_path(file_path: Path, columns_dtypes: dict[str, pl.DataType]) -> pl.DataFrame:
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
    exact_match: dict[str, list[str | int]] | None = None,
    fuzzy_match: dict[str, list[str]] | None = None,
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
    columns_to_read = list(set(list(exact_match.keys()) + list(fuzzy_match.keys()) if fuzzy_match else []))

    # Specify dtypes for columns
    dtypes = dict.fromkeys(columns_to_read, pl.Utf8)

    # Create scan object
    scan = pl.scan_csv(file_path, dtypes=dtypes, encoding="utf8-lossy", ignore_errors=True)

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


def query_column_by_search_terms(file_path: Path, column_name: str, search_terms: list[str]) -> pl.DataFrame:
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

    return scan.filter(pl.col(column_name).str.contains(combined_pattern, literal=False)).collect()


def query_columns_by_search_terms(
    file_path: Path,
    search_terms: list[str],
    columns: list[str] | None = None,
    case_sensitive: bool = False,
    escaped_terms: bool = False,
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
    if columns is None:
        columns = ["SEC_PARTY", "COLLATERAL"]

    # Specify dtypes for columns
    dtypes = dict.fromkeys(columns, pl.Utf8)

    if escaped_terms:
        # Escape special regex characters in search terms
        search_terms = [re.escape(term) for term in search_terms]

    if case_sensitive:
        combined_pattern = "|".join(f"(^|\\s){re.escape(term)}" for term in search_terms)
    else:
        combined_pattern = "(?i)" + "|".join(f"(^|\\s){re.escape(term)}" for term in search_terms)

    scan = pl.scan_csv(file_path, dtypes=dtypes, encoding="utf8-lossy", ignore_errors=True)

    # Build the filter expression for the single column
    filter_expr = pl.col("COLLATERAL").str.contains(
        combined_pattern,
        literal=False,
    )

    print(f"Filter expression: {filter_expr}")

    # Apply the filter and collect results
    result = scan.filter(filter_expr).collect()

    # Remove duplicates
    result = result.unique()

    print(f"Total rows found (after deduplication): {len(result)}")

    assert isinstance(result, pl.DataFrame)

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
        raise ValueError("Exactly two column names must be provided: [latitude, longitude]")

    # Specify dtypes for columns
    dtypes = dict.fromkeys(columns, pl.Float64)

    scan = pl.scan_csv(file_path, dtypes=dtypes, encoding="utf8-lossy", ignore_errors=True)

    # Build the filter expression for the coordinate pairs
    filter_expr = pl.lit(False)
    for lat, lon in coordinate_pairs:
        lat_condition = (pl.col(columns[0]) >= lat - tolerance) & (pl.col(columns[0]) <= lat + tolerance)
        lon_condition = (pl.col(columns[1]) >= lon - tolerance) & (pl.col(columns[1]) <= lon + tolerance)
        filter_expr |= lat_condition & lon_condition

    # Apply the filter and collect results
    result = scan.filter(filter_expr).collect()

    # Remove duplicates
    result = result.unique()
    print(f"Total rows found (after deduplication): {len(result)}")

    return result


def write_lazy_results_to_csv(
    results: pl.LazyFrame,
    output_path: Path,
    print_results: bool = True,
    overwrite: bool = True,
) -> None:
    """
    Write the query results to a CSV file.

    Args:
    results (pl.LazyFrame): The LazyFrame containing the query results
    output_path (Path): The path where the CSV file should be saved
    print_results (bool, default = True): Whether to print the results
    write_if_empty (bool, default = False): Whether to write the results if they are empty
    """
    # length_of_results = results.select(pl.len()).collect().item()
    # if not write_if_empty and length_of_results == 0:
    #     return

    if not output_path.parent.exists():
        output_path.parent.mkdir(parents=True, exist_ok=True, mode=0o777)

    if overwrite and output_path.exists():
        output_path.unlink()

    results.sink_csv(output_path)

    if print_results:
        print(f"Results written to {output_path}")


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
    if isinstance(search_string, str) and ("\\" in search_string or any(c in search_string for c in ".*+?{}[]()^$|")):
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
            cleaned_name = name.replace("(", "").replace(")", "").replace("[", "").replace("]", "").replace("*", "")
            # Escape remaining special regex characters
            escaped_name = re.escape(cleaned_name)
            processed_terms.append(escaped_name)

    # Handle empty result
    if not processed_terms:
        raise ValueError("No valid search terms provided")

    # Build final regex with word boundaries (using raw string for clarity)
    # Use non-capturing groups for efficiency
    regex_pattern = rf"(?:^|\b|\s)(?:{'|'.join(processed_terms)})(?:\b|\s|$)"

    return regex_pattern


def _format_search_string(
    search_string: str | list[str],
    convert_to_lowercase: bool = False,
    use_regex: bool = True,
) -> str:
    # If search_string is already formatted or contains regex patterns, return as is
    if isinstance(search_string, str) and ("\\" in search_string or any(c in search_string for c in ".*+?{}[]()^$|")):
        return search_string

    if convert_to_lowercase:
        search_string = [name.lower() for name in search_string]

    if len(search_string) <= 1:
        regex_search = f"(^|\\b|\\s| )({search_string})(\\b|\\s|$| )"
    elif len(search_string) > 1:
        regex_search = []
        for name in search_string:
            # Only clean and escape if the string doesn't contain regex patterns
            if not use_regex or not any(c in name for c in ".*+?{}[]()^$|"):
                # Clean and escape special characters
                cleaned_name = name.replace(")", "").replace("(", "").replace("[", "").replace("]", "").replace("*", "")
                # Escape special regex characters
                escaped_name = re.escape(cleaned_name)
                regex_search.append(escaped_name)
            else:
                # Keep regex patterns as is
                regex_search.append(name)

        # Add boundary conditions once for the entire pattern
        regex_search = f"(^|\\b|\\s| )({'|'.join(regex_search)})(\\b|\\s|$| )"

    return regex_search


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


def write_results_to_parquet(
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

    results.write_parquet(output_path)

    if print_results:
        print(f"Results written to {output_path}: \n {len(results)} rows")


def get_value_counts_lazy(csv_path: Path, column_name: str) -> pl.DataFrame:
    """
    Get the value counts for a specific column in a lazy DataFrame.

    Args:
    df (pl.DataFrame): The input DataFrame
    column_name (str): The name of the column to compute value counts for

    Returns:
    pl.DataFrame: A DataFrame containing the value counts for the specified column
    """
    return scan_csv_path(csv_path, {column_name: str}).select(pl.col(column_name).value_counts(sort=True)).collect()


def get_multi_column_value_counts_lazy(csv_path: Path, column_names: list[str]) -> pl.DataFrame:
    """
    Get the combined value counts for multiple specified columns in a lazy DataFrame.

    Args:
    csv_path (Path): The path to the CSV file
    column_names (list[str]): A list of column names to compute combined value counts for

    Returns:
    pl.DataFrame: A DataFrame containing the combined value counts for the specified columns
    """
    # Create a dictionary to store the schema for scan_csv
    schema = dict.fromkeys(column_names, str)

    # Scan the CSV file
    df = scan_csv_path(csv_path, schema)

    # Compute combined value counts for the specified columns
    return df.group_by(column_names).agg(pl.count().alias("count")).sort("count", descending=True).collect()


def format_vc_for_two_columns(value_counts: pl.DataFrame) -> pl.DataFrame:
    """
    Format the value counts DataFrame for two columns.

    Args:
    value_counts (pl.DataFrame): DataFrame containing the value counts

    Returns:
    pl.DataFrame: Formatted Polars DataFrame with two columns
    """
    df_value_counts = value_counts.to_pandas()

    df_value_counts["party"] = df_value_counts["SEC_PARTY"].apply(lambda x: x["SEC_PARTY"])

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
    schema = dict.fromkeys(join_columns, str)
    schema[party_column] = str

    # Scan the CSV file
    full_df_lazy = scan_csv_path(csv_path, schema)

    # Group the full DataFrame by join columns and aggregate party names into a list
    associated_names = full_df_lazy.group_by(join_columns).agg(pl.col(party_column).alias("temp_associated_names"))

    # Convert count_df to lazy
    count_df_lazy = count_df.lazy()

    # Join the count DataFrame with the associated names
    result = count_df_lazy.join(associated_names, on=join_columns, how="left")

    # Add the associated_names column
    result = result.with_columns([pl.col("temp_associated_names").fill_null([]).alias("associated_names_filled")])

    # Make the list unique
    result = result.with_columns([pl.col("associated_names_filled").list.unique().alias("associated_names")])

    # Drop the temporary columns
    result = result.drop(["temp_associated_names", "associated_names_filled"])

    return result.collect()


def query_by_year_range(
    file_path: Path,
    start_year: int,
    end_year: int,
    date_column: str = "UCC_FILING_DATE",
) -> pl.DataFrame:
    """
    Query the CSV file to find rows where the date column falls within the specified year range.

    Args:
    file_path (Path): Path to the CSV file
    start_year (int): Start year of the range (inclusive)
    end_year (int): End year of the range (inclusive)
    date_column (str): Name of the column containing the date (default: "UCC_FILING_DATE")

    Returns:
    pl.DataFrame: Filtered DataFrame containing rows within the specified year range
    """
    # Specify dtype for the date column
    dtypes = {date_column: pl.Utf8}

    scan = pl.scan_csv(file_path, dtypes=dtypes, encoding="utf8-lossy", ignore_errors=True)

    # Build the filter expression for the year range
    filter_expr = pl.col(date_column).str.strptime(pl.Date, "%Y-%m-%d").dt.year().is_between(start_year, end_year)

    # Apply the filter and collect results
    result = scan.filter(filter_expr).collect()

    print(f"Total rows found: {len(result)}")

    return result


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


def process_search_phrase(
    phrase: str | list[str] | None,
    exclude: bool = False,
    use_default_exclude: bool = True,
) -> str:
    """
    Process search phrase(s) to create a properly formatted regex pattern.

    Args:
        phrase (Optional[Union[str, list[str]]]): Input phrase or list of phrases
        exclude (bool): Whether this is for exclusion. Defaults to False.
        use_default_exclude (bool): Whether to use default exclusion terms. Defaults to True.

    Returns:
        str: Processed regex pattern
    """
    global EXCLUDED_SEARCH_TERMS
    DEFAULT_EXCLUDE_PHRASE = "(?i)PLAYSTATION|mower|DVD|WEEDEATER|SHOTGUN|FLATSCREEN|FLAT SCREEN|HAND GUN|9MM|9NUN"

    if exclude and use_default_exclude and isinstance(phrase, str):
        if phrase == "":
            return DEFAULT_EXCLUDE_PHRASE + "|".join(EXCLUDED_SEARCH_TERMS)
        # Skip the (?i) in DEFAULT_EXCLUDE_PHRASE
        return f"(?i){phrase}|{EXCLUDED_SEARCH_TERMS[4:]}"

    if isinstance(phrase, list) and all(isinstance(p, str) for p in phrase):
        return f"(?i){'| '.join(phrase)}"

    if isinstance(phrase, str) and phrase:
        return f"(?i){phrase}"

    return phrase or DEFAULT_EXCLUDE_PHRASE + "|".join(EXCLUDED_SEARCH_TERMS)


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

        # For literal matching, use any_horizontal with multiple contains
        # conditions = [
        #     pl.col(col_name).str.contains(term, literal=True) for term in terms
        # ]
        # return pl.any_horizontal(conditions)
    # For large lists, split into chunks
    chunk_conditions = []

    for i in range(0, len(terms), chunk_size):
        chunk = terms[i : i + chunk_size]

        pattern = format_search_string(chunk, convert_to_lowercase=not case_sensitive)

        # print("Processing for pattern")
        # rprint(pattern)
        # if use_regex:
        #     pattern = "|".join(pl.lit(term) for term in chunk)
        #     chunk_conditions.append(
        #         pl.col(col_name).str.contains(pattern, literal=False)
        #     )
        # else:
        #     # Use any_horizontal for each chunk
        #     sub_conditions = [
        #         pl.col(col_name).str.contains(term, literal=True) for term in chunk
        #     ]
        #     chunk_conditions.append(pl.any_horizontal(sub_conditions))

        chunk_conditions.append(lowercase_column.str.contains(pattern, literal=False))

    # Combine all chunks with OR logic
    return pl.any_horizontal(chunk_conditions)


def find_rows_with_phrase_df(
    df: pl.DataFrame | pl.LazyFrame,
    columns: list[str] | None = None,
    phrase: str | list[str] | None = "",
    exclude: bool = False,
    case_sensitive: bool = False,
    use_regex: bool = True,
    debug: bool = False,
    skip_garbage_collection: bool = True,
) -> pl.DataFrame | pl.LazyFrame:
    if exclude and not phrase:
        # This isn't necessarily a warning since sometimes we wouldn't have
        # any exclude terms, but can chain the methods together for convenience.
        if debug:
            print("Returning original without exclusion since no exclude terms provided.")
        return df

    if isinstance(df, pl.LazyFrame):
        # Store initial count for comparison
        initial_count = df.select(pl.count()).collect().item()
    else:
        initial_count = df.height

    # Convert to LazyFrame if not already
    lf = df.lazy() if isinstance(df, pl.DataFrame) else df

    # Validate columns exist
    if not columns:
        columns = ["COLLATERAL"]

    if exclude and not phrase:
        if debug:
            print("Returning original without exclusion since no exclude terms provided.")
        return df

    # Process the search phrase
    # processed_phrase = process_search_phrase(phrase, exclude, use_default_exclude)
    # processed_phrase = format_search_string(
    #     phrase, convert_to_lowercase=not case_sensitive
    # )

    result = lf

    if exclude:
        included_results_test = lf

        processed_phrase = format_search_string(phrase, convert_to_lowercase=not case_sensitive)

        for column in columns:
            included_results_test = included_results_test.filter(
                ~pl.col(column).str.to_lowercase().str.contains(processed_phrase)
            )

        result = included_results_test.collect()
    else:
        rows_to_keep = []
        # Keep rows that match the term in any column
        for column in columns:
            # rows_to_keep.append(
            #     pl.col(column).str.to_lowercase().str.contains(processed_phrase)
            # )

            # Create match condition for this column
            match_condition = create_substring_conditions(column, phrase, chunk_size=10, case_sensitive=case_sensitive)
            rows_to_keep.append(match_condition)

        result = result.filter(pl.any_horizontal(rows_to_keep))

    # Get final count and calculate excluded rows
    if isinstance(result, pl.LazyFrame):
        final_count = result.select(pl.count()).collect().item()
    else:
        final_count = result.height

    excluded_count = initial_count - final_count

    if exclude and excluded_count > 0:
        print(f"Excluded {excluded_count:,} rows ({(excluded_count / initial_count) * 100:.2f}% of total)")

        # If everything was excluded.
        if excluded_count == initial_count:
            print("No rows were found after exclusion. Returning original.")
            return df

        print(f"Excluded {excluded_count:,} rows ({(excluded_count / initial_count) * 100:.2f}% of total)")

    if not skip_garbage_collection and df is not None:
        # Force garbage collection
        # Set to None to remove the reference
        df = None
        # Force garbage collection
        import gc

        gc.collect()

    return result.collect() if isinstance(result, pl.LazyFrame) else result


def align_schema(df: pl.DataFrame, target_schema: dict, fill_missing_columns: bool = True) -> pl.DataFrame:
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
                df = df.with_columns(pl.col(col_name).str.replace_all(r"^\s*$", "0").cast(pl.Int64, strict=False))
            elif dtype in (pl.Float64, pl.Float32) and df[col_name].dtype == pl.String:
                df = df.with_columns(pl.col(col_name).str.replace_all(r"^\s*$", "0.0").cast(dtype, strict=False))
            else:
                df = df.with_columns(pl.col(col_name).cast(dtype))

        elif fill_missing_columns:
            df = df.with_columns(pl.Series(name=col_name, values=[None] * len(df), dtype=dtype))

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
            list(set(columns_to_search + ["FILE_DATE", "ROW_INDEX"] + (additional_columns or [])))
        )

    result = find_rows_with_phrase_df(
        df=scan_df,
        columns=columns_to_search,
        phrase=search_terms,
        use_regex=use_regex,
    )

    del scan_df

    # Only collect if explicitly requested
    if not lazy and hasattr(result, "collect"):
        return result.collect()

    return result


def show_excluded_rows(
    original_df: pl.DataFrame | pl.LazyFrame, filtered_df: pl.DataFrame | pl.LazyFrame
) -> pl.DataFrame:
    """
    Shows the rows that were excluded during filtering.

    Args:
        original_df: The original LazyFrame before exclusion filtering
        filtered_df: The LazyFrame after exclusion filtering

    Returns:
        DataFrame containing only the rows that were excluded
    """
    # Get unique identifiers from both dataframes
    # Using anti_join to find rows in original_df that aren't in filtered_df
    # We need to ensure we have a unique identifier for each row

    # First collect both dataframes to compute the difference
    # For large datasets, we might want a more memory-efficient approach
    # but this is the most straightforward solution

    if isinstance(original_df, pl.LazyFrame):
        original_collected = original_df.collect()
    else:
        original_collected = original_df

    if isinstance(filtered_df, pl.LazyFrame):
        filtered_collected = filtered_df.collect()
    else:
        filtered_collected = filtered_df

    # Find the rows that exist in original but not in filtered
    # This assumes your dataframe has some columns that can uniquely identify each row
    # If you don't have a natural primary key, you might need to create one

    # Get all column names to use for comparison
    all_columns = original_collected.columns

    # Use anti_join to find rows in original that aren't in filtered
    return original_collected.join(filtered_collected, on=all_columns, how="anti")


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
    """
    DuckDB text search with same logic as Polars version.

    Key optimizations:
    - Single regex pass with (term1|term2|...) pattern
    - Pushdown filtering during parquet read
    - No Python iteration over rows
    """

    if exclude and not search_terms:
        if debug:
            print("No exclude terms provided, returning all rows")
        # Return full file
        conn = duckdb.connect()
        return conn.execute("SELECT * FROM read_parquet(?)", [str(fpath)])

    # Normalize search terms to list
    if isinstance(search_terms, str):
        search_terms = [search_terms]

    # Default column
    if not columns_to_search:
        columns_to_search = ["COLLATERAL"]

    # Build column list
    # Column selection
    if read_all_columns:
        select_cols = "*"
    else:
        cols = list(set(columns_to_search + ["FILE_DATE", "ROW_INDEX"] + (additional_columns or [])))
        select_cols = ", ".join(cols)

    conn = duckdb.connect()

    # Build regex pattern
    # Build WHERE conditions
    # if use_regex:
    #     combined_pattern = ("(?i)" if not case_sensitive else "") + "|".join(
    #         search_terms
    #     )
    #     conditions = [f"regexp_matches({col}, ?)" for col in columns_to_search]
    #     params = [str(fpath)] + [combined_pattern] * len(columns_to_search)
    # else:
    #     conditions = []
    #     params = [str(fpath)]

    #     for col in columns_to_search:
    #         if case_sensitive:
    #             col_conditions = [f"{col} LIKE ?" for _ in search_terms]
    #             params.extend([f"%{term}%" for term in search_terms])
    #         else:
    #             col_conditions = [f"contains(lower({col}), ?)" for _ in search_terms]
    #             params.extend([term.lower() for term in search_terms])
    #         conditions.append(f"({' OR '.join(col_conditions)})")
    # Build WHERE conditions
    if use_regex:
        # Add word boundaries to each term
        if word_boundary:
            bounded_terms = [rf"\b{term}\b" for term in search_terms]
        else:
            bounded_terms = search_terms

        combined_pattern = ("(?i)" if not case_sensitive else "") + "|".join(bounded_terms)
        conditions = [f"regexp_matches({col}, ?)" for col in columns_to_search]
        params = [str(fpath)] + [combined_pattern] * len(columns_to_search)
    else:
        # Literal matching with word boundaries
        conditions = []
        params = [str(fpath)]

        for col in columns_to_search:
            if word_boundary:
                # Match whole words: (^|\\s)term(\\s|$)
                # Using regexp for literal with boundaries is more reliable
                if case_sensitive:
                    # DuckDB: use regexp_matches with escaped literals
                    col_conditions = [f"regexp_matches({col}, ?)" for _ in search_terms]
                    # Escape special regex chars and add boundaries
                    import re

                    params.extend([rf"\b{re.escape(term)}\b" for term in search_terms])
                else:
                    col_conditions = [f"regexp_matches({col}, ?)" for _ in search_terms]
                    import re

                    params.extend([rf"(?i)\b{re.escape(term)}\b" for term in search_terms])
            else:
                # Original substring matching
                if case_sensitive:
                    col_conditions = [f"{col} LIKE ?" for _ in search_terms]
                    params.extend([f"%{term}%" for term in search_terms])
                else:
                    col_conditions = [f"contains(lower({col}), ?)" for _ in search_terms]
                    params.extend([term.lower() for term in search_terms])

            conditions.append(f"({' OR '.join(col_conditions)})")

    where_clause = " OR ".join(conditions) if not exclude else " AND ".join([f"NOT ({cond})" for cond in conditions])

    query = f"""
    SELECT {select_cols}
    FROM read_parquet(?)
    WHERE {where_clause}
    """

    # Build params
    if use_regex:
        params = [str(fpath)] + [combined_pattern] * len(columns_to_search)
    else:
        # For literal: need term per column
        params = [str(fpath)]
        for _ in columns_to_search:
            if case_sensitive:
                params.extend([f"%{term}%" for term in search_terms])
            else:
                params.extend([term.lower() for term in search_terms])

    result = conn.execute(query, params)
    # Drop helper column
    return result.pl()


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
    use_regex: bool = True,
    debug: bool = False,
) -> tuple[pl.DataFrame | pl.DataFrame]:
    """
    Efficiently search through partitioned parquet files for specific terms.

    This function optimizes for performance with large datasets by:
    1. Only scanning required partitions
    2. Using lazy evaluation and predicate pushdown
    3. Only loading necessary columns
    4. Applying filters at scan time when possible
    5. Utilizing parallel processing with joblib

    Args:
        base_path: Path to the partitioned parquet directory
        search_terms: List of terms to search for
        columns_to_search: Columns to search within
        partition_by: The column used for partitioning
        partition_values: List of partition values to read. If None, reads all partitions.
        exclude_terms: Optional list of terms to exclude from results
        additional_columns: Additional columns to include in results beyond search columns
        lazy: If True, returns a LazyFrame; if False, returns a DataFrame
        n_jobs: Number of parallel jobs to run (-1 means using all processors)

    Returns:
        A LazyFrame or DataFrame containing the matching rows
    """
    base_path = Path(base_path)

    # Determine columns to load
    columns_to_load = list(set(columns_to_search + (additional_columns or [])))

    # If no partition values specified, discover all available partitions
    if partition_values is None:
        partition_values = []
        for path in base_path.glob(f"{partition_by}=*"):
            if path.is_dir():
                try:
                    value = path.name.split("=")[1]
                    # Try to convert to int if possible
                    with contextlib.supress(ValueError):
                        value = int(value)
                    partition_values.append(value)
                except IndexError:
                    continue

    # Build paths for each partition to read
    paths_to_read = []
    for value in partition_values:
        partition_dir = base_path / f"{partition_by}={value}"
        if partition_dir.exists():
            # Get all parquet files in this partition
            parquet_files = list(partition_dir.rglob("*.parquet"))
            paths_to_read.extend([str(p) for p in parquet_files])

    if debug:
        rprint(sorted(paths_to_read))

    if not paths_to_read:
        raise ValueError(f"No parquet files found for the specified partition values: {partition_values}")

    if use_duckdb:
        # results_list_raw = [
        #     find_rows_with_phrase_duckdb(
        #         fpath=f,
        #         search_terms=search_terms,
        #         columns_to_search=columns_to_search,
        #         case_sensitive=False,
        #         additional_columns=additional_columns,
        #         debug=debug,
        #     )
        #     for f in tqdm(paths_to_read)
        # ]

        # Process files in parallel
        def process_file(path):
            return find_rows_with_phrase_duckdb(
                fpath=path,
                search_terms=search_terms,
                columns_to_search=columns_to_search,
                case_sensitive=False,
                additional_columns=additional_columns,
                debug=debug,
            )

        # Use joblib for parallel processing
        results_list_raw = Parallel(n_jobs=n_jobs)(delayed(process_file)(path) for path in paths_to_read)

    else:
        # Process files in parallel
        def process_file(path):
            return find_rows_with_phrase_from_fpath(
                fpath=path,
                search_terms=search_terms,
                columns_to_search=columns_to_search,
                lazy=False,  # We need to collect for parallel processing
                additional_columns=additional_columns,
                read_all_columns=read_all_columns,
                use_regex=use_regex,
            )

        # Use joblib for parallel processing
        results_list_raw = Parallel(n_jobs=n_jobs)(delayed(process_file)(path) for path in paths_to_read)

    # Make sure that the column orders is aligned.
    if read_all_columns:
        # Then look within the schemas of the results_list_raw and get the union of all columns
        all_columns = {col for df in results_list_raw for col in df.columns}
        target_schema = dict.fromkeys(all_columns, pl.Utf8)
    else:
        columns_to_load = list(set(columns_to_load + (additional_columns or [])))
        target_schema = target_schema or dict.fromkeys(columns_to_load, pl.Utf8)

    results_list = [align_schema(df, target_schema=target_schema) for df in results_list_raw]

    # Combine results efficiently
    if results_list:
        combined_results = pl.concat(results_list, how="vertical_relaxed")

        # Apply exclusion filter if needed and capture excluded rows
        if exclude_terms:
            # Get rows that match exclusion terms (to be excluded)
            excluded_rows = find_rows_with_phrase_df(
                df=combined_results,
                columns=columns_to_search,
                phrase=exclude_terms,
                exclude=False,  # Get matching rows
            )

            # Get rows that don't match exclusion terms (filtered result)
            combined_results = find_rows_with_phrase_df(
                df=combined_results,
                columns=columns_to_search,
                phrase=exclude_terms,
                exclude=True,  # Exclude matching rows
            )
        else:
            # No exclusion terms - create empty dataframe with same schema
            excluded_rows = combined_results.head(0)
    else:
        # Handle case where results_list is empty
        combined_results = pl.DataFrame()
        excluded_rows = pl.DataFrame()

    return combined_results, excluded_rows
