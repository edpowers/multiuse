"""Display output functions for Polars."""

import re
from datetime import datetime

import altair as alt
import polars as pl
from rich import print as rprint


def plot_product_sales(
    df: pl.DataFrame,
    date_col: str = "FILE_DATE",
    product_col: str = "product_name",
    quantity_col: str = "quantity",  # Ensure this matches your actual CSV column
    top_n: int = 10,
    title: str = "Product Sales Over Time",
) -> alt.Chart:
    # 1. Prepare: Create 'month' column and ensure quantity is numeric
    #    We fill null quantities with 0 (or 1, depending on your business logic)
    df_clean = df.with_columns(
        [
            pl.col(date_col).cast(pl.Date).dt.truncate("1mo").alias("month"),
            pl.col(quantity_col).fill_null(0).alias("qty_safe"),
        ],
    )

    # 2. Aggregate by Month (not daily date)
    df_agg = (
        df_clean.group_by(["month", product_col])
        .agg(pl.col("qty_safe").sum().alias("total_quantity"))
        .sort("month")
    )

    # 3. Identify Top N Products by Volume
    top_products = (
        df_agg.group_by(product_col)
        .agg(pl.col("total_quantity").sum())
        .top_k(top_n, by="total_quantity")
        .get_column(product_col)
    )

    # 4. Filter & Plot
    df_filtered = df_agg.filter(pl.col(product_col).is_in(top_products.implode()))

    return (
        alt.Chart(df_filtered.to_pandas())
        .mark_line(point=True)
        .encode(
            # Now 'month' actually exists in the dataframe
            x=alt.X("month:T", title="Date", axis=alt.Axis(format="%Y-%b")),
            y=alt.Y("total_quantity:Q", title="Total Quantity Sold"),
            color=alt.Color(f"{product_col}:N", title="Product"),
            tooltip=[
                alt.Tooltip("month:T", format="%Y-%B"),
                product_col,
                alt.Tooltip("total_quantity", format=","),
            ],
        )
        .properties(width=800, height=400, title=title)
        .interactive()
    )


def plot_transaction_timeline(
    df: pl.DataFrame,
    date_col: str = "FILE_DATE",
    start_date: str | None = None,
    end_date: str | None = None,
    width: int = 900,
    height: int = 500,
    title: str | None = None,
    freq: str = "quarter",  # "quarter" or "month"
    quantity_column: str | None = None,
) -> alt.Chart:
    """
    Plot transaction counts over time with improved readability.

    Args:
        df: DataFrame with date column
        date_col: Column name containing dates
        start_date: Optional filter start (format: %Y-%m-%d)
        end_date: Optional filter end (format: %Y-%m-%d)
        width: Chart width
        height: Chart height
        title: Chart title
        freq: Aggregation frequency ("quarter" or "month")
    """
    # Parse dates
    plot_df = df.select(pl.col(date_col).str.to_date("%Y-%m-%d").alias("date"))

    if quantity_column and quantity_column in plot_df:
        count_expression = pl.col(quantity_column).sum().alias("count")
    else:
        count_expression = pl.len().alias("count")

    # Filter date range
    if start_date:
        plot_df = plot_df.filter(
            pl.col("date") >= datetime.strptime(start_date, "%Y-%m-%d"),
        )
    if end_date:
        plot_df = plot_df.filter(
            pl.col("date") <= datetime.strptime(end_date, "%Y-%m-%d"),
        )

    # Aggregate by period
    if freq == "quarter":
        plot_df = plot_df.with_columns(
            [
                pl.col("date").dt.year().alias("year"),
                pl.col("date").dt.quarter().alias("quarter"),
            ],
        )

        counts = (
            plot_df.group_by(["year", "quarter"])
            .agg(count_expression)
            .sort(["year", "quarter"])
            .with_columns(period=pl.format("{}Q{}", "year", "quarter"))
        )

        x_type = "period:N"
        x_title = "Quarter"
        axis_config = alt.Axis(labelAngle=-45, labelOverlap=False)

    else:  # month
        plot_df = plot_df.with_columns(
            period=pl.col("date").dt.strftime("%Y-%m-01").str.to_date(),
        )

        counts = plot_df.group_by("period").agg(count_expression).sort("period")

        x_type = "period:T"
        x_title = "Month"
        axis_config = alt.Axis(
            labelAngle=-45,
            format="%b %Y",
            labelOverlap=False,
            tickCount="month",
        )

    # Create chart
    return (
        alt.Chart(counts)
        .mark_line(point=alt.OverlayMarkDef(size=60, filled=True), strokeWidth=2.5)
        .encode(
            x=alt.X(x_type, title=x_title, axis=axis_config),
            y=alt.Y(
                "count:Q",
                title="Transaction Count",
                scale=alt.Scale(zero=True),
            ),
            tooltip=[
                alt.Tooltip(x_type, title="Period"),
                alt.Tooltip("count:Q", title="Count", format=","),
            ],
        )
        .properties(
            width=width,
            height=height,
            title=alt.TitleParams(
                text=title or "Transaction Volume Over Time",
                fontSize=16,
                anchor="middle",
            ),
        )
        .configure_axis(
            labelFontSize=11,
            titleFontSize=13,
            grid=True,
        )
        .configure_view(strokeWidth=0)
    )


def highlight_results(
    results,
    search_terms: list[str],
    exclude_terms: list[str] | None = None,
    columns_to_search: list[str] | None = None,
    sample_size: int = 125,
    show_all: bool = False,
    print_output: bool = True,
    return_highlighted_results: bool = False,
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

    # For all search terms, remove any special characters:
    search_terms = list(map(lambda x: re.sub(r"[\^-]", "", x), search_terms))  # noqa: C417

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

    if return_highlighted_results:
        return highlighted_results

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
            .alias(f"{col}_highlighted"),
        )

    if exclude_terms:
        for col in columns:
            df = df.with_columns(
                pl.col(col)
                .map_elements(
                    lambda x: highlight_terms(str(x), exclude_terms, exclude_color),
                    return_dtype=pl.Utf8,
                )
                .alias(f"{col}_highlighted"),
            )
    return df


def print_random_samples(
    results: pl.DataFrame,
    column: str = "COLLATERAL",
    n: int = 5,
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
