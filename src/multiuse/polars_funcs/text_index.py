"""Persistent, APPEND-ONLY inverted index for large-scale fuzzy UCC search.

The problem: every ticker search re-scans the whole multi-million-row corpus with
regex. Instead we maintain a reusable inverted index that is **built incrementally as
new partition paths arrive** (no rebuilds), and answer fuzzy term queries cheaply:

    index (as data lands):  per new partition file -> INSERT into the year's index DB
        docs(ROW_INDEX, FILE_DATE, <text cols>)   -- row text, for verification
        postings(token, ROW_INDEX)                 -- inverted index (append-only)
        vocab(token)                               -- distinct tokens, for fuzzy expand
    query (fast):  fuzzy-expand each term word against the vocabulary (vocab-scale)
        -> postings give candidate ROW_INDEXes (no corpus scan)
        -> trigram CONTAINMENT verify on the small candidate set

Design notes:
- One ``.duckdb`` per ``FILE_YEAR`` under ``index_dir``: bounds DB size, lets queries
  open only the relevant years, and matches the FILE_YEAR/FILE_MONTH partition layout.
- Everything is append-only: a new partition just INSERTs its rows/tokens. There is no
  index rebuild step, so ``index_partition(path)`` slots directly into the existing
  "get new partition paths" workflow.
- Tokenization is a digit-preserving split (``[^a-z0-9]+``) so product tokens like
  ``excelsiusgps`` / ``creo`` / ``s2ai`` / ``3d`` survive; the same split is used for
  query terms. Tokens shorter than 3 chars are not indexed (no usable trigram).
- Pure DuckDB SQL + the shared n-gram macros; no extensions.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import duckdb
import polars as pl

from multiuse.polars_funcs.polars_analysis_funcs import _create_native_jaccard_macros

DEFAULT_INDEX_COLUMNS: tuple[str, ...] = ("COLLATERAL", "SEC_PARTY")
_SPLIT_RE = "[^a-z0-9]+"  # digit-preserving tokenizer (keeps "3d", "s2ai", ...)
_TOKEN_RE = re.compile(r"[0-9a-z]+")
_MIN_TOKEN_CHARS = 3  # need >= 3 chars for a trigram
_YEAR_RE = re.compile(r"FILE_YEAR=(\d+)")


def _year_db_path(index_dir: str | Path, year: int) -> Path:
    return Path(index_dir) / f"ucc_index_{int(year)}.duckdb"


def _manifest_path(index_dir: str | Path) -> Path:
    return Path(index_dir) / "index_manifest.json"


def _read_manifest(index_dir: str | Path) -> dict:
    p = _manifest_path(index_dir)
    return json.loads(p.read_text()) if p.exists() else {}


def _write_manifest(index_dir: str | Path, manifest: dict) -> None:
    _manifest_path(index_dir).write_text(json.dumps(manifest, indent=2, sort_keys=True))


def _year_files(base_path: str | Path, year: int) -> list[str]:
    return sorted(str(p) for p in Path(base_path).glob(f"FILE_YEAR={year}/**/*.parquet"))


def _year_of(path: str | Path) -> int:
    m = _YEAR_RE.search(str(path))
    if not m:
        ve = f"Could not infer FILE_YEAR from path: {path}"
        raise ValueError(ve)
    return int(m.group(1))


def _connect(db_path: Path, memory_limit: str, threads: int, *, read_only: bool) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = duckdb.connect(str(db_path), read_only=read_only)
    conn.execute(f"PRAGMA memory_limit='{memory_limit}'")
    conn.execute(f"PRAGMA threads={int(threads)}")
    return conn


def _ensure_schema(conn: duckdb.DuckDBPyConnection, columns: tuple[str, ...]) -> None:
    col_defs = ", ".join(f'"{c}" VARCHAR' for c in columns)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS docs "
        f"(ROW_INDEX BIGINT, FILE_DATE VARCHAR, {col_defs})",
    )
    conn.execute("CREATE TABLE IF NOT EXISTS postings (token VARCHAR, ROW_INDEX BIGINT)")
    conn.execute("CREATE TABLE IF NOT EXISTS vocab (token VARCHAR PRIMARY KEY)")
    # Persist the n-gram / jaccard macros so read-only queries can call them.
    _create_native_jaccard_macros(conn)


def _append_one(conn: duckdb.DuckDBPyConnection, path: str, columns: tuple[str, ...]) -> None:
    """Append a single partition file's rows + tokens (no rebuild)."""
    col_q = ", ".join(f'"{c}"' for c in columns)
    conn.execute(
        f"INSERT INTO docs SELECT CAST(ROW_INDEX AS BIGINT), "
        f"CAST(FILE_DATE AS VARCHAR), {col_q} FROM read_parquet(?)",
        [path],
    )
    token_union = " UNION ALL ".join(
        f"SELECT ri, UNNEST(string_split_regex(lower(\"{c}\"), '{_SPLIT_RE}')) AS tok "
        f"FROM src"
        for c in columns
    )
    conn.execute(
        f"""
        CREATE OR REPLACE TEMP TABLE _toks AS
        WITH src AS (
            SELECT CAST(ROW_INDEX AS BIGINT) AS ri, {col_q} FROM read_parquet(?)
        )
        SELECT DISTINCT tok, ri FROM ({token_union})
        WHERE length(tok) >= {_MIN_TOKEN_CHARS}
        """,
        [path],
    )
    conn.execute("INSERT INTO postings SELECT tok, ri FROM _toks")
    conn.execute("INSERT OR IGNORE INTO vocab SELECT DISTINCT tok FROM _toks")
    conn.execute("DROP TABLE _toks")


def _append_files_to_db(
    db_path: Path,
    files: list[str],
    columns: tuple[str, ...],
    memory_limit: str,
    threads: int,
) -> None:
    conn = _connect(db_path, memory_limit, threads, read_only=False)
    try:
        _ensure_schema(conn, columns)
        for f in files:
            _append_one(conn, f, columns)
    finally:
        conn.close()


def build_index(
    base_path: str | Path,
    year: int,
    index_dir: str | Path,
    *,
    columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS,
    memory_limit: str = "6GB",
    threads: int = 4,
    overwrite: bool = False,
) -> Path:
    """Build the index for one ``FILE_YEAR`` from all its partition files (append-only).

    Equivalent to creating an empty year DB and ``index_partition``-ing every file.
    Memory- and thread-capped; processes one file at a time. Returns the year DB path.
    """
    files = _year_files(base_path, year)
    if not files:
        ve = f"No parquet files found for FILE_YEAR={year} under {base_path}"
        raise ValueError(ve)

    db_path = _year_db_path(index_dir, year)
    if db_path.exists():
        if not overwrite:
            ve = f"Index already exists at {db_path}; pass overwrite=True to rebuild."
            raise FileExistsError(ve)
        db_path.unlink()

    _append_files_to_db(db_path, files, columns, memory_limit, threads)

    manifest = _read_manifest(index_dir)
    manifest[str(year)] = {"columns": list(columns), "files": files}
    _write_manifest(index_dir, manifest)
    return db_path


def index_partition(
    parquet_path: str | Path,
    index_dir: str | Path,
    *,
    columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS,
    memory_limit: str = "6GB",
    threads: int = 4,
) -> Path:
    """Incrementally index ONE partition file -- the build-as-you-go hook.

    Infers ``FILE_YEAR`` from the path, appends the file's rows/tokens to that year's
    index DB (creating it if needed), and records it in the manifest. Idempotent: a path
    already in the manifest is skipped. No rebuild of existing data.

    Call this for each new partition path your ingestion workflow produces.
    """
    path = str(parquet_path)
    year = _year_of(path)
    db_path = _year_db_path(index_dir, year)

    manifest = _read_manifest(index_dir)
    entry = manifest.setdefault(str(year), {"columns": list(columns), "files": []})
    if path in entry["files"]:
        return db_path

    _append_files_to_db(db_path, [path], columns, memory_limit, threads)

    entry["files"] = sorted({*entry["files"], path})
    _write_manifest(index_dir, manifest)
    return db_path


def append_files(
    base_path: str | Path,
    year: int,
    index_dir: str | Path,
    new_files: list[str | Path] | None = None,
    *,
    columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS,
    memory_limit: str = "6GB",
    threads: int = 4,
) -> Path:
    """Append new partition files for a year (append-only; discovers all if not given)."""
    db_path = _year_db_path(index_dir, year)
    manifest = _read_manifest(index_dir)
    entry = manifest.setdefault(str(year), {"columns": list(columns), "files": []})

    candidates = (
        [str(f) for f in new_files] if new_files is not None else _year_files(base_path, year)
    )
    pending = [f for f in candidates if f not in set(entry["files"])]
    if not pending:
        return db_path

    _append_files_to_db(db_path, pending, columns, memory_limit, threads)
    entry["files"] = sorted({*entry["files"], *pending})
    _write_manifest(index_dir, manifest)
    return db_path


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _query_one_year(
    db_path: Path,
    terms_df: pl.DataFrame,
    words_df: pl.DataFrame,
    columns: tuple[str, ...],
    similarity_threshold: float,
    expand_threshold: float,
    fuzzy: bool,
    memory_limit: str,
    threads: int,
) -> pl.DataFrame:
    conn = _connect(db_path, memory_limit, threads, read_only=True)
    try:
        conn.register("_q_terms", terms_df)
        conn.register("_q_words", words_df)

        # 1+2. Fuzzy-expand query words to vocabulary tokens (vocab-scale trigram join).
        if fuzzy:
            conn.execute(
                f"""
                CREATE TEMP TABLE _expanded AS
                WITH qg AS (
                    SELECT word, UNNEST(_lsh_ngrams(word, 3)) AS g FROM _q_words
                ),
                qsize AS (SELECT word, COUNT(DISTINCT g) AS qn FROM qg GROUP BY word),
                vg AS (SELECT token, UNNEST(_lsh_ngrams(token, 3)) AS g FROM vocab),
                shared AS (
                    SELECT qg.word, vg.token, COUNT(DISTINCT qg.g) AS k
                    FROM qg JOIN vg ON qg.g = vg.g
                    GROUP BY qg.word, vg.token
                )
                SELECT DISTINCT shared.token
                FROM shared JOIN qsize USING (word)
                WHERE shared.k::DOUBLE / qsize.qn >= {float(expand_threshold)}
                """,
            )
        else:
            conn.execute(
                "CREATE TEMP TABLE _expanded AS "
                "SELECT DISTINCT word AS token FROM _q_words",
            )

        # 3. Postings -> candidate ROW_INDEXes (no corpus scan).
        conn.execute(
            """
            CREATE TEMP TABLE _cand AS
            SELECT DISTINCT p.ROW_INDEX
            FROM postings p
            WHERE p.token IN (SELECT token FROM _expanded)
            """,
        )

        # 4+5. Trigram CONTAINMENT verify on candidate rows only.
        doc_blocks = [
            f"""
                SELECT e.ROW_INDEX, q.term, '{col}' AS matched_column, q.g
                FROM (
                    SELECT cs.ROW_INDEX, UNNEST(_lsh_ngrams(lower(cs."{col}"), 3)) AS g
                    FROM docs cs JOIN _cand USING (ROW_INDEX)
                ) e
                JOIN q ON e.g = q.g
                """
            for col in columns
        ]
        doc_union = "\n                UNION ALL\n".join(doc_blocks)
        col_out = ", ".join(f'cs."{c}"' for c in columns)
        verify_sql = f"""
            WITH q AS (
                SELECT term, term_lc, UNNEST(_lsh_ngrams(term_lc, 3)) AS g FROM _q_terms
            ),
            qsize AS (SELECT term, COUNT(DISTINCT g) AS qn FROM q GROUP BY term),
            shared AS (
                SELECT ROW_INDEX, term, matched_column, COUNT(DISTINCT g) AS k
                FROM ({doc_union})
                GROUP BY ROW_INDEX, term, matched_column
            ),
            scored AS (
                SELECT shared.ROW_INDEX, shared.term AS matched_term,
                       shared.matched_column, shared.k::DOUBLE / qsize.qn AS lsh_score
                FROM shared JOIN qsize ON shared.term = qsize.term
                WHERE shared.k::DOUBLE / qsize.qn >= {float(similarity_threshold)}
            )
            SELECT cs.ROW_INDEX, cs.FILE_DATE, {col_out},
                   scored.matched_term, scored.matched_column, scored.lsh_score
            FROM scored JOIN docs cs ON scored.ROW_INDEX = cs.ROW_INDEX
            QUALIFY row_number() OVER (
                PARTITION BY cs.ROW_INDEX ORDER BY scored.lsh_score DESC
            ) = 1
        """
        return conn.execute(verify_sql).pl()
    finally:
        conn.close()


def query_index(
    index_dir: str | Path,
    search_terms: str | list[str],
    years: int | list[int],
    *,
    columns: tuple[str, ...] = DEFAULT_INDEX_COLUMNS,
    similarity_threshold: float = 0.7,
    expand_threshold: float = 0.7,
    fuzzy: bool = True,
    min_word_chars: int = 4,
    exclude_terms: list[str] | None = None,
    memory_limit: str = "4GB",
    threads: int = 4,
) -> pl.DataFrame:
    """Query the index for rows fuzzily matching any of ``search_terms``.

    Fuzzy-expands the terms' words against each year's vocabulary, uses the postings to
    get candidate ROW_INDEXes (no corpus scan), then verifies with trigram containment.
    Returns one row per matching ROW_INDEX with its best match:
    ``ROW_INDEX, FILE_DATE, <columns>, matched_term, matched_column, lsh_score``.
    """
    if isinstance(search_terms, str):
        search_terms = [search_terms]
    if isinstance(years, int):
        years = [years]
    if not search_terms:
        return pl.DataFrame()

    terms_lc = [str(t).lower() for t in search_terms]
    terms_df = pl.DataFrame({"term": search_terms, "term_lc": terms_lc})
    words = sorted({
        w for t in terms_lc for w in _tokenize(t) if len(w) >= min_word_chars
    })
    if not words:  # all terms shorter than min_word_chars -> fall back to raw terms
        words = sorted({t for t in terms_lc if t})
    words_df = pl.DataFrame({"word": words})

    frames = []
    for year in years:
        db_path = _year_db_path(index_dir, year)
        if not db_path.exists():
            continue
        frames.append(
            _query_one_year(
                db_path, terms_df, words_df, columns,
                similarity_threshold, expand_threshold, fuzzy, memory_limit, threads,
            ),
        )

    if not frames:
        return pl.DataFrame()
    combined = pl.concat(frames, how="vertical_relaxed")
    if combined.is_empty():
        return combined
    combined = (
        combined.sort("lsh_score", descending=True)
        .unique(subset=["ROW_INDEX"], keep="first")
    )

    if exclude_terms:
        pattern = "|".join(re.escape(e.lower()) for e in exclude_terms if e)
        if pattern:
            mask = pl.lit(value=False)
            for col in columns:
                if col in combined.columns:
                    mask = mask | (
                        pl.col(col).str.to_lowercase().str.contains(pattern)
                    )
            combined = combined.filter(~mask)

    return combined
