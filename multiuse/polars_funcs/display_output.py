"""Display output functions for Polars."""

import re

import polars as pl
from rich import print as rprint


def highlight_results(
    results,
    search_terms: list[str],
    exclude_terms: list[str] | None = None,
    columns_to_search: list[str] | None = None,
    sample_size: int = 125,
    show_all: bool = False,
    print_output: bool = True,
):
    """
    Highlight and optionally print search results from a Polars DataFrame.

    Args
    ----
    results (pl.DataFrame):
        The DataFrame containing the search results.
    search_terms (list):
        List of search terms used to highlight the results.
    columns_to_search (list, default = ["COLLATERAL", "SEC_PARTY"]):
        List of column names to search and highlight.
    sample_size (int, default = 125):
        Number of rows to sample if not showing all.
    show_all (bool, default = False):
        If True, show all results instead of a sample.
    print_output (bool, default = True):
        If True, print the highlighted results.

    Returns:
    -------
    pl.DataFrame:
        The DataFrame with highlighted columns added.
    """
    if not columns_to_search:
        columns_to_search = ["COLLATERAL", "SEC_PARTY"]

    highlighted_results = highlight_columns(
        df=results.sample(min(sample_size, results.height)),
        columns=columns_to_search,
        search_terms=search_terms,
        exclude_terms=exclude_terms,
    )

    if not show_all and len(highlighted_results) > sample_size:
        sample = highlighted_results.sample(sample_size)
    else:
        sample = highlighted_results

    if print_output:
        for row in sample.iter_rows():
            for col, value in zip(highlighted_results.columns, row, strict=False):
                if col.endswith("_highlighted"):
                    try:
                        rprint(f"{col}:\n{value}\n")
                    except Exception as e:
                        print(f"Error printing {col}: {e}")
            rprint("-" * 50)

        print(f"Total rows found - {len(results)}")

    return results


# Example usage:
# highlighted_df = highlight_results(results, columns_to_search, search_terms, sample_size=100, show_all=False, print_output=True)


def highlight_terms(text: str, search_terms: list[str], color: str = "red") -> str:
    """
    Highlight matching terms in the given text using Rich markup, including partial matches.

    Args:
    text (str): The text to search in.
    search_terms (list): List of terms to highlight.
    color (str): Color to use for highlighting. Default is 'red'.

    Returns:
    str: Text with highlighted terms using Rich markup.
    """
    # Sort search terms by length (longest first) to avoid nested highlights
    search_terms = sorted(search_terms, key=len, reverse=True)

    for term in search_terms:
        # Remove leading/trailing whitespace and escape special regex characters
        cleaned_term = re.escape(term)
        # Use a regex pattern with word boundaries to match the term
        pattern = re.compile(f"(?i)({cleaned_term})")

        # Apply highlighting using Rich's markup
        text = pattern.sub(f"[{color}]\\1[/{color}]", text)

    return text


def highlight_columns(
    df: pl.DataFrame,
    columns: list[str],
    search_terms: list[str],
    exclude_terms: list[str] | None = None,
    color: str = "yellow",
    exclude_color: str = "bright_red",
) -> pl.DataFrame:
    """
    Apply highlighting to specified columns in a Polars DataFrame.

    Args:
    df (pl.DataFrame): The DataFrame to process.
    columns (list): List of column names to apply highlighting to.
    search_terms (list): List of terms to highlight.
    color (str): Color to use for highlighting. Default is 'yellow'.

    Returns:
    pl.DataFrame: DataFrame with highlighted text in specified columns.
    """
    for col in columns:
        df = df.with_columns(
            pl.col(col)
            .map_elements(
                lambda x: highlight_terms(str(x), search_terms, color),
                return_dtype=pl.Utf8,
            )
            .alias(f"{col}_highlighted")
        )

    if exclude_terms:
        for col in columns:
            df = df.with_columns(
                pl.col(col)
                .map_elements(
                    lambda x: highlight_terms(str(x), exclude_terms, exclude_color),
                    return_dtype=pl.Utf8,
                )
                .alias(f"{col}_highlighted")
            )
    return df


def print_random_samples(
    results: pl.DataFrame, column: str = "COLLATERAL", n: int = 5
) -> pl.DataFrame:
    """
    Select n random rows from a specified column in the results DataFrame and print them.

    Args:
    results (pl.DataFrame): The input DataFrame
    column (str): The name of the column to sample from
    n (int): The number of random samples to print (default: 5)

    Returns:
    None
    """
    random_sample = results.sample(n=n)
    random_sample_data = random_sample.select(column)
    list_of_data = random_sample_data.to_pandas().squeeze().tolist()

    for i, data in enumerate(list_of_data, 1):
        print()
        rprint(f"{i}. {data}")
        print()

    return random_sample
