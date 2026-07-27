import contextlib
import re
from functools import partial
from itertools import batched
from pathlib import Path
from typing import Literal, cast

import duckdb
import polars as pl
from joblib import Parallel, delayed
from rapidfuzz import fuzz
from rich import print as rprint
from tqdm import tqdm

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
    initial_count = collect_df(df).height
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
    excluded_count = initial_count - result_final.height

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


LSH_METADATA_COLUMNS = ("matched_term", "matched_column", "lsh_score")


def _try_load_lsh_extension(conn: duckdb.DuckDBPyConnection) -> bool:
    """Try to INSTALL/LOAD the community ``lsh`` extension.

    Returns True when the extension is available for the running DuckDB build,
    False otherwise (e.g. no prebuilt binary exists for this DuckDB version).
    The first successful install fetches the binary over the network and caches
    it under ``~/.duckdb/extensions``; subsequent loads are offline.
    """
    try:
        conn.execute("INSTALL lsh FROM community;")
        conn.execute("LOAD lsh;")
    except duckdb.Error:
        return False
    return True


def _create_native_jaccard_macros(conn: duckdb.DuckDBPyConnection) -> None:
    """Register extension-free char-n-gram Jaccard macros on the connection.

    ``_lsh_ngrams(s, n)`` returns the set (distinct list) of character n-grams of
    ``s``; ``_lsh_jaccard(a, b)`` returns the Jaccard similarity of two such sets.
    These reproduce the community extension's ``lsh_jaccard`` scores exactly while
    requiring no third-party extension.
    """
    conn.execute(
        """
        CREATE OR REPLACE MACRO _lsh_ngrams(s, n) AS
            list_distinct(list_transform(
                range(1, length(s) - n + 2),
                i -> substring(s, CAST(i AS INTEGER), CAST(n AS INTEGER))
            ));
        """,
    )
    conn.execute(
        """
        CREATE OR REPLACE MACRO _lsh_jaccard(a, b) AS
            CASE
                WHEN a IS NULL OR b IS NULL THEN NULL
                WHEN len(list_distinct(list_concat(a, b))) = 0 THEN 0.0
                ELSE len(list_intersect(a, b))::DOUBLE
                     / len(list_distinct(list_concat(a, b)))
            END;
        """,
    )


def _lsh_row_select_clause(base_cols: list[str], read_all_columns: bool) -> str:
    if read_all_columns:
        return "r.*"
    return ", ".join(f'r."{c}"' for c in base_cols)


def _build_native_lsh_query(
    fpath: Path | str,
    columns_to_search: list[str],
    base_cols: list[str],
    ngram_width: int,
    threshold: float,
    case_sensitive: bool,
    read_all_columns: bool,
) -> tuple[str, list[str]]:
    """Extension-free query: char-n-gram Jaccard, cross-join + threshold filter.

    Exact and complete (no LSH recall loss), at O(terms x rows) per file -- fine
    for moderate files and small term lists; n-grams are materialised once per
    side so only the cheap set comparison repeats.
    """
    n = int(ngram_width)
    row_select = _lsh_row_select_clause(base_cols, read_all_columns)
    # Build per-column ngram projections (lowercased unless case-sensitive).
    rg_projections = []
    for i, col in enumerate(columns_to_search):
        col_expr = f'r."{col}"' if case_sensitive else f'lower(r."{col}")'
        rg_projections.append(f"_lsh_ngrams({col_expr}, {n}) AS rg_{i}")
    rg_select = ", ".join(rg_projections)
    exclude_clause = ", ".join(f"rg_{i}" for i in range(len(columns_to_search)))

    blocks = []
    for i, col in enumerate(columns_to_search):
        blocks.append(
            f"""
            SELECT _r.* EXCLUDE ({exclude_clause}),
                   _t.term AS matched_term,
                   '{col}' AS matched_column,
                   _lsh_jaccard(_r.rg_{i}, _t.tg) AS lsh_score
            FROM _r CROSS JOIN _t
            WHERE _lsh_jaccard(_r.rg_{i}, _t.tg) >= {threshold}
            """,
        )
    union_sql = "\n            UNION ALL\n".join(blocks)

    query = f"""
        WITH _t AS MATERIALIZED (
            SELECT term, term_lc, _lsh_ngrams(term_lc, {n}) AS tg FROM terms
        ),
        _r AS MATERIALIZED (
            SELECT {row_select}, {rg_select}
            FROM read_parquet(?) r
        )
        SELECT * FROM (
            {union_sql}
        )
        QUALIFY row_number() OVER (
            PARTITION BY ROW_INDEX ORDER BY lsh_score DESC
        ) = 1
    """
    return query, [str(fpath)]


def _build_lsh_extension_query(
    fpath: Path | str,
    columns_to_search: list[str],
    base_cols: list[str],
    ngram_width: int,
    band_size: int,
    seeds: tuple[int, ...],
    threshold: float,
    case_sensitive: bool,
    read_all_columns: bool,
) -> tuple[str, list[str]]:
    """MinHash-LSH query: band-hash equijoin (blocking) + exact Jaccard filter.

    One band per seed (``lsh_min(..., 1, band_size, seed)[1]``); rotating the seed
    across UNION-ed blocks adds bands, raising recall. Blocking is a hash equijoin
    so candidate generation is sub-quadratic; ``lsh_jaccard`` then re-scores exactly.
    """
    n = int(ngram_width)
    bsize = int(band_size)
    row_select = _lsh_row_select_clause(base_cols, read_all_columns)

    blocks: list[str] = []
    params: list[str] = []
    for col in columns_to_search:
        col_expr = f'r."{col}"' if case_sensitive else f'lower(r."{col}")'
        for seed in seeds:
            s = int(seed)
            blocks.append(
                f"""
                SELECT {row_select},
                       t.term AS matched_term,
                       '{col}' AS matched_column,
                       lsh_jaccard({col_expr}, t.term_lc, {n}) AS lsh_score
                FROM read_parquet(?) r
                JOIN terms t
                  ON lsh_min({col_expr}, {n}, 1, {bsize}, {s})[1]
                   = lsh_min(t.term_lc, {n}, 1, {bsize}, {s})[1]
                WHERE lsh_jaccard({col_expr}, t.term_lc, {n}) >= {threshold}
                """,
            )
            params.append(str(fpath))
    union_sql = "\n                UNION ALL\n".join(blocks)

    query = f"""
        SELECT * FROM (
            {union_sql}
        )
        QUALIFY row_number() OVER (
            PARTITION BY ROW_INDEX ORDER BY lsh_score DESC
        ) = 1
    """
    return query, params


def _containment_anchor_pattern(
    search_terms,
    case_sensitive=False,
    min_anchor_chars=4,
    window=5,
    step=3,
):
    token_re = re.compile(r"[0-9a-z]+" if not case_sensitive else r"[0-9A-Za-z]+")
    anchors, seen = [], set()

    def add_windows(word):
        if len(word) <= window:
            cands = [word]
        else:
            starts = list(range(0, len(word) - window + 1, step))
            if starts[-1] != len(word) - window:
                starts.append(len(word) - window)
            cands = [word[s : s + window] for s in starts]
        for c in cands:
            if c not in seen:
                seen.add(c)
                anchors.append(c)

    for term in search_terms:
        tokens = token_re.findall(term if case_sensitive else term.lower())
        if not tokens:
            continue
        sized = [t for t in tokens if len(t) >= min_anchor_chars]
        for tok in sized if sized else [max(tokens, key=len)]:
            add_windows(tok)
    if not anchors:
        return None
    return "|".join(re.escape(a) for a in anchors)


_VIN_RUN = r"[A-Z0-9]{17}"  # aligned pure run (primary)
_VIN_SPLIT = (
    r"[A-Z0-9]{5,16}[ /'.-][A-Z0-9]{1,12}"  # one internal separator (indel/split)
)


def _extract_vin_tokens(
    fpath: Path | str,
    columns_to_search: list[str],
    additional_columns: list[str] | None = None,
) -> pl.DataFrame:
    """One linear, term-independent pass: pull VIN-shaped tokens + keys per file.

    Emits aligned 17-char alnum runs plus single-separator splits (so an indel
    that dropped one char is still captured), one row per (source row, token).
    """
    keep = list(dict.fromkeys(["ROW_INDEX", "FILE_DATE", *(additional_columns or [])]))
    sel = ", ".join(f'"{c}"' for c in keep)
    col = columns_to_search[0]
    q = f"""
        SELECT {sel}, unnest(list_concat(
                   regexp_extract_all(upper("{col}"), $run),
                   regexp_extract_all(upper("{col}"), $split))) AS tok
        FROM read_parquet($f)
    """
    return (
        duckdb.connect()
        .execute(
            q,
            {"run": _VIN_RUN, "split": _VIN_SPLIT, "f": str(fpath)},
        )
        .pl()
    )


def _canon_vin_expr(c: str) -> pl.Expr:
    """Definitional-only canonicalization: strip non-alnum, I->1 O->0 Q->0."""
    return (
        pl.col(c)
        .str.replace_all(r"[^A-Z0-9]", "")
        .str.replace_all("I", "1")
        .str.replace_all("O", "0")
        .str.replace_all("Q", "0")
    )


def _resolve_tokens_to_vins(
    df_tok: pl.DataFrame,
    vins: list[str],
    threshold: float = 0.88,
    win: int = 5,
    step: int = 3,
    len_lo: int = 15,
    len_hi: int = 19,
) -> pl.DataFrame:
    """Match extracted tokens against the clean VIN set. Two tiers:

    Tier 1 exact after canonicalization (clean + I/O/Q, score 1.0); Tier 2 blocks
    the residual by shared char-windows and verifies with edit-distance ratio.
    Returns (VIN, filing)-grain rows with matched_term (=VIN_REF) and lsh_score.
    """
    if df_tok.is_empty() or not vins:
        return pl.DataFrame()
    df = df_tok.with_columns(_canon_vin_expr("tok").alias("tc")).filter(
        pl.col("tc").str.len_chars().is_between(len_lo, len_hi),
    )
    vinset = set(vins)
    exact = df.filter(pl.col("tc").is_in(vinset)).with_columns(
        pl.col("tc").alias("matched_term"),
        pl.lit(1.0).alias("lsh_score"),
    )

    block: dict[str, list[str]] = {}
    for v in vins:
        for s in sorted({*range(0, len(v) - win + 1, step), len(v) - win}):
            block.setdefault(v[s : s + win], []).append(v)

    cutoff = threshold * 100.0
    hits = []
    for row in df.filter(~pl.col("tc").is_in(vinset)).iter_rows(named=True):
        tc = row["tc"]
        cands = {
            v for s in range(len(tc) - win + 1) for v in block.get(tc[s : s + win], ())
        }
        if not cands:
            continue
        best = max(cands, key=lambda v: fuzz.ratio(tc, v))
        score = fuzz.ratio(tc, best)
        if score >= cutoff:
            hits.append({**row, "matched_term": best, "lsh_score": score / 100.0})
    fuzzy = pl.DataFrame(hits) if hits else exact.head(0)

    return (
        pl.concat([exact, fuzzy], how="diagonal_relaxed")
        .sort("lsh_score", descending=True)
        .unique(["ROW_INDEX", "matched_term"], keep="first")
        .sort("ROW_INDEX")
    )


# def _containment_anchor_pattern(
#     search_terms: list[str],
#     case_sensitive: bool,
#     min_anchor_chars: int = 4,
# ) -> str | None:
#     """Regex alternation of each term's significant words, for fast candidate pruning.

#     Collects every alphanumeric token of length >= ``min_anchor_chars`` from each term
#     (falling back to the term's longest token if it has none). A row is a candidate only
#     if it contains at least one such word verbatim. Because a term scoring above a high
#     containment threshold must share most of its n-grams, a genuine mention almost always
#     contains one of the term's words spelled correctly -- so this prunes the ~99% of rows
#     that mention nothing while keeping recall high even when a *different* word is typo'd.
#     Returns None if no usable token exists (caller then skips pruning).
#     """
#     token_re = re.compile(r"[0-9a-z]+" if not case_sensitive else r"[0-9A-Za-z]+")
#     anchors: list[str] = []
#     seen: set[str] = set()
#     for term in search_terms:
#         tokens = token_re.findall(term if case_sensitive else term.lower())
#         if not tokens:
#             continue
#         sized = [t for t in tokens if len(t) >= min_anchor_chars]
#         chosen = sized if sized else [max(tokens, key=len)]
#         for tok in chosen:
#             if tok not in seen:
#                 seen.add(tok)
#                 anchors.append(tok)
#     if not anchors:
#         return None
#     return "|".join(re.escape(a) for a in anchors)


def _build_native_containment_query(
    fpath: Path | str,
    columns_to_search: list[str],
    base_cols: list[str],
    ngram_width: int,
    threshold: float,
    case_sensitive: bool,
    read_all_columns: bool,
    anchor_pattern: str | None,
) -> tuple[str, list[str]]:
    """Extension-free CONTAINMENT query for finding short terms in LONG text, fast.

    Scores ``|ngrams(term) intersect ngrams(doc)| / |ngrams(term)|`` -- how much of the
    *term* appears in the document, independent of document length. Right for "is this
    term (fuzzily) mentioned in this collateral text?", where symmetric Jaccard is tiny.

    Two stages, so cost scales with the (tiny) number of *matching* rows, not the corpus:
      1. **Block**: a single vectorized ``regexp_matches`` against ``anchor_pattern``
         (one representative literal per term) prunes to candidate rows, reading only
         the search columns. This eliminates the ~99% of rows that mention nothing.
      2. **Score**: explode each *candidate's* n-grams and hash-join them against the
         query-term n-grams, counting shared n-grams per (row, term).
    Without the block stage, exploding n-grams for every row is what makes whole-corpus
    fuzzy search slow; with it, the trigram work touches only plausible rows.
    If ``anchor_pattern`` is None the block stage is skipped (slower, max recall).
    """
    n = int(ngram_width)
    row_select = _lsh_row_select_clause(base_cols, read_all_columns)

    # Lowercased (unless case-sensitive) search-column copies used for matching.
    sc_aliases = []
    for i, col in enumerate(columns_to_search):
        expr = f'r."{col}"' if case_sensitive else f'lower(r."{col}")'
        sc_aliases.append(f"{expr} AS _c{i}")
    sc_select = ", ".join(sc_aliases)
    exclude_clause = ", ".join(f"_c{i}" for i in range(len(columns_to_search)))

    params: list[str] = [str(fpath)]
    if anchor_pattern:
        where_clause = "WHERE " + " OR ".join(
            f"regexp_matches(_c{i}, ?)" for i in range(len(columns_to_search))
        )
        params += [anchor_pattern] * len(columns_to_search)
    else:
        where_clause = ""

    doc_blocks = []
    for i, col in enumerate(columns_to_search):
        doc_blocks.append(
            f"""
            SELECT e.ROW_INDEX, q.term, '{col}' AS matched_column, q.g
            FROM (
                SELECT cand.ROW_INDEX, du.g
                FROM cand CROSS JOIN UNNEST(_lsh_ngrams(cand._c{i}, {n})) AS du(g)
            ) e
            JOIN q ON e.g = q.g
            """,
        )
    doc_union = "\n            UNION ALL\n".join(doc_blocks)

    query = f"""
        WITH cand AS MATERIALIZED (
            SELECT {row_select}, {sc_select}
            FROM read_parquet(?) r
            {where_clause}
        ),
        q AS (
            SELECT term, term_lc, u.g
            FROM terms CROSS JOIN UNNEST(_lsh_ngrams(term_lc, {n})) AS u(g)
        ),
        qsize AS (SELECT term, COUNT(DISTINCT g) AS qn FROM q GROUP BY term),
        shared AS (
            SELECT ROW_INDEX, term, matched_column, COUNT(DISTINCT g) AS k
            FROM ({doc_union})
            GROUP BY ROW_INDEX, term, matched_column
        ),
        scored AS (
            SELECT shared.ROW_INDEX,
                   shared.term AS matched_term,
                   shared.matched_column,
                   shared.k::DOUBLE / qsize.qn AS lsh_score
            FROM shared JOIN qsize ON shared.term = qsize.term
            WHERE shared.k::DOUBLE / qsize.qn >= {threshold}
        )
        SELECT cand.* EXCLUDE ({exclude_clause}),
               scored.matched_term, scored.matched_column, scored.lsh_score
        FROM scored JOIN cand ON scored.ROW_INDEX = cand.ROW_INDEX
        QUALIFY row_number() OVER (
            PARTITION BY cand.ROW_INDEX ORDER BY scored.lsh_score DESC
        ) = 1
    """
    return query, params


def find_rows_with_phrase_lsh(
    fpath: Path | str,
    search_terms: str | list[str],
    columns_to_search: list[str] | None = None,
    *,
    metric: Literal["jaccard", "containment"] = "jaccard",
    similarity_threshold: float = 0.7,
    ngram_width: int | None = None,
    band_size: int = 2,
    seeds: tuple[int, ...] = (1, 2, 3, 4),
    backend: Literal["auto", "native", "lsh_extension"] = "auto",
    prefilter: bool = True,
    min_anchor_chars: int = 4,
    read_all_columns: bool = False,
    additional_columns: list[str] | None = None,
    case_sensitive: bool = False,
    return_match_metadata: bool = True,
    debug: bool = False,
) -> pl.DataFrame:
    """Fuzzy name/entity search over a parquet file using char-n-gram similarity.

    Locality-sensitive analogue of :func:`find_rows_with_phrase_duckdb`. Instead of
    substring/regex containment, each search term is compared to each row's text by
    **character n-gram similarity**, and rows scoring at least ``similarity_threshold``
    are returned. This catches spelling/spacing variants and typos
    (``"excelsius gps"`` ~ ``"excelsiusgps"``) that exact matching misses.

    Pick the ``metric`` to match the column you are searching:
        * ``"jaccard"`` (default): symmetric whole-string similarity,
          ``|A intersect B| / |A union B|``. Best for *name-like* columns
          (e.g. ``SEC_PARTY``) and dedup -- "are these two strings the same entity?".
          Does NOT find a short term inside a long field (a short query against a long
          document yields a tiny Jaccard) and does not resolve acronyms
          (``"Envista CU"`` ~ ``"Envista Credit Union"`` ~ 0.40).
        * ``"containment"``: asymmetric, ``|ngrams(term) intersect ngrams(doc)| /
          |ngrams(term)|`` -- how much of the *term* appears in the document,
          independent of document length. Use this to find terms (fuzzily) **mentioned
          inside long free text** like ``COLLATERAL``. Always native. By default it
          runs **block-then-score** (see ``prefilter``): a fast substring filter prunes
          to candidate rows, then trigram scoring runs only on those, so cost scales
          with the number of *matching* rows rather than the corpus (~0.3s vs ~30s on
          1M rows of diverse long text in benchmarks).

    Backends (apply to ``metric="jaccard"`` only):
        * ``"native"`` / ``"auto"`` (default): extension-free char-n-gram similarity in
          pure DuckDB SQL. Version-proof and memory-safe on NULL/empty/short/unicode
          text. Jaccard runs O(terms x rows) per file.
        * ``"lsh_extension"``: true MinHash-LSH via the DuckDB community ``lsh``
          extension (``lsh_min`` band blocking + ``lsh_jaccard``). **Opt-in only**: a
          small third-party C++ extension pinned to exact DuckDB versions that has been
          observed to segfault on real data inside parallel workers, so ``"auto"`` never
          selects it. ``metric="containment"`` ignores ``backend`` (always native).

    Args:
        fpath: Path to the parquet file.
        search_terms: Term or terms to fuzzy-match against the columns.
        columns_to_search: Columns to compare against. Defaults to ``["SEC_PARTY"]``
            (use e.g. ``["COLLATERAL"]`` with ``metric="containment"`` for long text).
        metric: ``"jaccard"`` (names/dedup) or ``"containment"`` (terms in long text).
        similarity_threshold: Minimum similarity (0-1) for a match.
        ngram_width: Character n-gram width. Defaults to 2 (bigrams) for ``"jaccard"``
            and 3 (trigrams) for ``"containment"`` if not set.
        band_size: MinHashes AND-ed per band (extension backend); higher -> more
            precision, less recall.
        seeds: One band per seed (extension backend); more seeds -> more recall.
        backend: ``"auto"``, ``"native"``, or ``"lsh_extension"`` (jaccard only).
        prefilter: ``"containment"`` only. If True (default), prune to candidate rows
            with a fast substring filter (each term's words as anchors) before n-gram
            scoring -- the key to performance on large corpora. Set False for an
            exhaustive scan (slower; only needed if a term's every word may be typo'd).
        min_anchor_chars: ``"containment"`` only. Minimum word length used as a prefilter
            anchor (default 4). Raise it to drop common short words and pass fewer
            candidates (faster); lower it for higher recall.
        read_all_columns: Return all columns instead of the minimal set.
        additional_columns: Extra columns to include in the output.
        case_sensitive: If False (default), compare lowercased text.
        return_match_metadata: Append ``matched_term``, ``matched_column``,
            ``lsh_score`` columns describing the best match per row.
        debug: Print a one-line summary of the resolved plan (no SQL dump).

    Returns:
        A polars DataFrame of matching rows (one row per input row, keyed on
        ``ROW_INDEX``, keeping its single highest-scoring match).
    """
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    if not columns_to_search:
        columns_to_search = ["SEC_PARTY"]
    if not search_terms:
        return pl.DataFrame()
    if not seeds:
        ve = "seeds must contain at least one value"
        raise ValueError(ve)
    if not 0.0 <= similarity_threshold <= 1.0:
        ve = "similarity_threshold must be between 0.0 and 1.0"
        raise ValueError(ve)

    # Resolve n-gram width: bigrams for jaccard (names), trigrams for containment
    # (more selective on long text), unless the caller overrides.
    n = (
        ngram_width
        if ngram_width is not None
        else (3 if metric == "containment" else 2)
    )

    base_cols = list(
        set(
            columns_to_search + ["FILE_DATE", "ROW_INDEX"] + (additional_columns or []),
        ),
    )

    conn = duckdb.connect()
    try:
        # Resolve execution plan:
        #   metric="containment" -> native trigram inverted-equijoin (always); the
        #     extension has no containment primitive, so backend is ignored here.
        #   metric="jaccard"     -> native cross-join, or the opt-in lsh extension.
        # "auto" never selects the extension: it is third-party C++ that has segfaulted
        # on real data inside parallel workers (an uncatchable crash, so it cannot be
        # tried-then-recovered) and is pinned to exact DuckDB versions.
        use_extension = False
        if metric == "jaccard" and backend == "lsh_extension":
            use_extension = _try_load_lsh_extension(conn)
            if not use_extension:
                re_msg = (
                    "backend='lsh_extension' requested but the community 'lsh' "
                    "extension could not be loaded for this DuckDB version. Use "
                    "backend='auto' or 'native' (identical scores, no extension), "
                    "or pin a DuckDB version that has an lsh build."
                )
                raise RuntimeError(re_msg)
        if not use_extension:
            _create_native_jaccard_macros(conn)

        # Register the query terms as a small in-memory relation.
        term_lc = (
            pl.col("term") if case_sensitive else pl.col("term").str.to_lowercase()
        )
        terms_df = pl.DataFrame(
            {"term": [str(t) for t in search_terms]},
        ).with_columns(term_lc.alias("term_lc"))
        conn.register("terms", terms_df)

        if metric == "containment":
            anchor_pattern = (
                _containment_anchor_pattern(
                    search_terms,
                    case_sensitive,
                    min_anchor_chars,
                )
                if prefilter
                else None
            )
            plan = f"native/containment prefilter={'on' if anchor_pattern else 'off'}"
            query, params = _build_native_containment_query(
                fpath=fpath,
                columns_to_search=columns_to_search,
                base_cols=base_cols,
                ngram_width=n,
                threshold=float(similarity_threshold),
                case_sensitive=case_sensitive,
                read_all_columns=read_all_columns,
                anchor_pattern=anchor_pattern,
            )
        elif use_extension:
            plan = "lsh_extension/jaccard"
            query, params = _build_lsh_extension_query(
                fpath=fpath,
                columns_to_search=columns_to_search,
                base_cols=base_cols,
                ngram_width=n,
                band_size=band_size,
                seeds=seeds,
                threshold=float(similarity_threshold),
                case_sensitive=case_sensitive,
                read_all_columns=read_all_columns,
            )
        else:
            plan = "native/jaccard"
            query, params = _build_native_lsh_query(
                fpath=fpath,
                columns_to_search=columns_to_search,
                base_cols=base_cols,
                ngram_width=n,
                threshold=float(similarity_threshold),
                case_sensitive=case_sensitive,
                read_all_columns=read_all_columns,
            )

        if debug:
            print(
                f"[find_rows_with_phrase_lsh] plan={plan} ngram={n} "
                f"threshold={similarity_threshold} terms={len(search_terms)} "
                f"columns={columns_to_search} file={Path(fpath).name}",
            )

        result = conn.execute(query, params).pl()
    finally:
        conn.close()

    if not return_match_metadata:
        result = result.drop(list(LSH_METADATA_COLUMNS), strict=False)

    return result


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
    search_method: Literal["regex", "lsh", "extract"] = "regex",
    lsh_kwargs: dict | None = None,
    extract_kwargs: dict | None = None,
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
    if search_method == "lsh":
        # Print ONE summary here instead of forwarding debug into every per-file
        # worker (which prints an identical line per parquet file). Per-file debug
        # is still available by setting lsh_kwargs={"debug": True}.
        if debug:
            lk = lsh_kwargs or {}
            print(
                f"[search_partitioned_parquet] lsh/{lk.get('metric', 'jaccard')} over "
                f"{len(paths_to_read)} files, {len(search_terms)} terms, "
                f"columns={columns_to_search}, "
                f"threshold={lk.get('similarity_threshold', 0.7)}",
            )
        process_fn = partial(
            find_rows_with_phrase_lsh,
            search_terms=search_terms,
            columns_to_search=columns_to_search,
            additional_columns=additional_columns,
            read_all_columns=read_all_columns,
            **(lsh_kwargs or {}),
        )
    elif search_method == "extract":
        base = Path(base_path)
        cols = columns_to_search or ["COLLATERAL"]
        pvals = partition_values
        if pvals is None:
            pvals = [
                int(p.name.split("=")[1])
                if p.name.split("=")[1].isdigit()
                else p.name.split("=")[1]
                for p in base.glob(f"{partition_by}=*")
                if p.is_dir()
            ]
        paths = [
            str(f)
            for v in pvals
            for f in (base / f"{partition_by}={v}").rglob("*.parquet")
        ]
        if not paths:
            ve = f"No parquet files found for partition values: {pvals}"
            raise ValueError(ve)

        fn = partial(
            _extract_vin_tokens,
            columns_to_search=cols,
            additional_columns=additional_columns,
        )
        frames = Parallel(n_jobs=n_jobs)(delayed(fn)(p) for p in paths)
        df_tok = (
            pl.concat(
                [f for f in frames if f.height],
                how="diagonal_relaxed",
            )
            if any(f.height for f in frames)
            else pl.DataFrame()
        )

        results = _resolve_tokens_to_vins(
            df_tok,
            search_terms or [],
            **(extract_kwargs or {}),
        )
        return results, results.head(0)
    elif use_duckdb:
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

    # Preserve LSH match-metadata columns (and keep lsh_score numeric).
    schema_for_align: dict = dict(target_schema or {})
    if search_method == "lsh" and (lsh_kwargs or {}).get(
        "return_match_metadata",
        True,
    ):
        schema_for_align["matched_term"] = pl.Utf8
        schema_for_align["matched_column"] = pl.Utf8
        schema_for_align["lsh_score"] = pl.Float64

    results_list = [
        align_schema(df, target_schema=schema_for_align) for df in results_list_raw
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


def rejoin_with_original_row_indices(
    results_path: Path,
    search_path: Path,
    df_prod_min: pl.DataFrame,
    debug: bool = False,
) -> pl.DataFrame:
    # results_path = company_search.f_historical_for_row_index_mapping
    # search_path: company_search.partitioned_parquet_path

    if results_path is not None and results_path.exists():
        results_prev = pl.read_parquet(results_path)

        row_indices_find = (
            df_prod_min.filter(
                ~pl.col("ROW_INDEX")
                .cast(pl.Int64)
                .is_in(results_prev["ROW_INDEX"].cast(pl.Int64).implode()),
            )["ROW_INDEX"]
            .unique()
            .drop_nulls()
            .to_list()
        )
    else:
        row_indices_find = df_prod_min["ROW_INDEX"].unique().drop_nulls().to_list()
        results_prev = pl.DataFrame()

    if row_indices_find:
        if debug:
            rprint(f"Trying to find {len(row_indices_find)=}")

        # Will need to re-search based on the values defined - so as to pull in the other columns.
        results, _ = search_partitioned_parquet(
            base_path=search_path,
            search_terms=row_indices_find,
            exclude_terms=[],
            columns_to_search=["ROW_INDEX"],
            partition_by="FILE_YEAR",
            partition_values=list(range(2015, 2050)),
            search_by_indices=True,
            use_duckdb=True,
            debug=False,
            read_all_columns=True,
            batch_size=100,
            display_vc_counts=False,
            # use_regex=False,
        )
    else:
        results = results_prev

    if results_prev.height > 0 and row_indices_find:
        results = (
            pl.concat([results, results_prev], how="diagonal_relaxed")
            .unique(["ROW_INDEX"])
            .sort(["FILE_DATE"])
        )

    results = results.with_columns(pl.col("ROW_INDEX").cast(pl.Int64)).drop(
        [
            "SECADR1",
            "SECADR2",
            "SECTYPE",
            "DEBTTYPE",
            "COLLATERAL",
            "SEC_LONGITUDE",
            "SEC_LATITUDE",
            "load_date",
        ],
        strict=False,
    )

    results.write_parquet(results_path)

    return results


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
    search_method: Literal["regex", "lsh", "index", "extract"] = "regex",
    lsh_kwargs: dict | None = None,
    extract_kwargs: dict | None = None,
    index_dir: str | Path | None = None,
    index_kwargs: dict | None = None,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """
    Efficiently search through partitioned parquet files for specific terms.

    New params:
        batch_size: If provided, processes search_terms in batches of this size
        dedup_columns: Columns to use for deduplication (default: ["ROW_INDEX"])
        search_method: "regex" (default, substring/regex containment), "lsh"
            (fuzzy char-n-gram similarity via find_rows_with_phrase_lsh), or "index"
            (fuzzy lookup against a prebuilt FTS index via text_index.query_index --
            avoids re-scanning the corpus; requires index_dir).
        lsh_kwargs: Extra keyword args forwarded to find_rows_with_phrase_lsh when
            search_method="lsh" (e.g. {"similarity_threshold": 0.8, "backend": "auto"}).
        index_dir: Directory of per-year FTS index DBs (search_method="index").
        index_kwargs: Extra kwargs forwarded to text_index.query_index
            (e.g. {"similarity_threshold": 0.7, "expand_threshold": 0.7, "fuzzy": True}).
    """
    dedup_columns = dedup_columns or ["ROW_INDEX"]

    # Index path: bypass per-file scanning entirely -- query the prebuilt index.
    if search_method == "index":
        if index_dir is None:
            ve = "search_method='index' requires index_dir"
            raise ValueError(ve)
        from multiuse.polars_funcs.text_index import query_index

        results = query_index(
            index_dir=index_dir,
            search_terms=search_terms or [],
            years=partition_values or [],
            columns=tuple(columns_to_search or ("COLLATERAL", "SEC_PARTY")),
            exclude_terms=exclude_terms,
            **(index_kwargs or {}),
        )
        excluded = results.head(0)
        return results, excluded

    # Default to 50 as batch size if search terms > 50.
    if search_terms and len(search_terms) > batch_size_constant:
        batch_size = batch_size_constant

    # If batching requested, process in batches
    if batch_size and search_terms and len(search_terms) > batch_size:
        all_results = []
        all_excluded = []

        for batch in tqdm(
            batched(search_terms, batch_size),
            total=int(len(search_terms) / batch_size),
        ):
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
                search_method=search_method,
                lsh_kwargs=lsh_kwargs,
                extract_kwargs=extract_kwargs,
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
        search_method=search_method,
        lsh_kwargs=lsh_kwargs,
        extract_kwargs=extract_kwargs,
    )
