"""Column type inference: numeric / categorical / datetime / ID / text.

Inference is a suggestion shown to the user for confirmation, so every result
carries a human-readable `reason`. Runs on the full dataset before the split:
it only looks at column shapes and fits nothing.
"""

import re
from dataclasses import dataclass
from enum import StrEnum

import pandas as pd


class ColumnType(StrEnum):
    NUMERIC = "numeric"
    CATEGORICAL = "categorical"
    DATETIME = "datetime"
    ID = "id"
    TEXT = "text"


# Name patterns that suggest an identifier: "id", "user_id", "patient_nbr", "Unnamed: 0"
# (the last is how pandas names a saved index column with an empty header).
ID_NAME = re.compile(r"(^|_)(id|nbr)$|^id_|^unnamed: \d+$", re.IGNORECASE)

MIN_ROWS_FOR_UNIQUENESS = 20  # below this, "all values unique" says little
ID_NAME_MIN_UNIQUE_FRAC = 0.5  # an ID-named column must be mostly unique
PARSE_FRAC = 0.9  # share of text values that must parse as numbers/dates
TEXT_MIN_MEAN_LENGTH = 30  # free text: long strings...
TEXT_MIN_UNIQUE_FRAC = 0.5  # ...that rarely repeat
NEAR_EMPTY_FRAC = 0.95  # missing share at which a column is suggested for exclusion


@dataclass
class ColumnInfo:
    name: str
    type: ColumnType
    reason: str
    n_unique: int
    missing_frac: float


def infer_schema(df: pd.DataFrame) -> dict[str, ColumnInfo]:
    """Infer a type for every column, keyed by column name."""
    return {name: infer_column(df[name]) for name in df.columns}


def infer_column(s: pd.Series) -> ColumnInfo:
    present = s.dropna()
    n_unique = present.nunique()
    col_type, reason = _infer_type(str(s.name), present, n_unique)
    missing_frac = float(s.isna().mean()) if len(s) else 0.0
    return ColumnInfo(str(s.name), col_type, reason, int(n_unique), missing_frac)


def suggest_exclusions(schema: dict[str, ColumnInfo]) -> dict[str, str]:
    """Columns we suggest dropping before analysis, mapped to the reason.

    The caller should not offer the chosen target column for exclusion.
    """
    suggestions = {}
    for name, info in schema.items():
        if info.missing_frac >= NEAR_EMPTY_FRAC:
            suggestions[name] = f"near-empty ({info.missing_frac:.0%} missing)"
        elif info.n_unique <= 1:
            suggestions[name] = "constant (one distinct value)"
        elif info.type == ColumnType.ID:
            suggestions[name] = "ID-like (no predictive meaning)"
    return suggestions


def _infer_type(name: str, present: pd.Series, n_unique: int) -> tuple[ColumnType, str]:
    """Rules are checked in order; the first match wins."""
    n = len(present)
    unique_frac = n_unique / n if n else 0.0

    if pd.api.types.is_bool_dtype(present):
        return ColumnType.CATEGORICAL, "boolean values"
    if pd.api.types.is_datetime64_any_dtype(present):
        return ColumnType.DATETIME, "already parsed as dates"
    if ID_NAME.search(name) and unique_frac >= ID_NAME_MIN_UNIQUE_FRAC:
        return ColumnType.ID, f"name looks like an ID and {unique_frac:.0%} of values are unique"

    if pd.api.types.is_numeric_dtype(present):
        if _is_row_counter(present):
            return ColumnType.ID, "consecutive unique integers (looks like a row number)"
        return ColumnType.NUMERIC, "numeric values"

    # Everything else is text-like: decide what the strings represent.
    text = present.astype(str)
    if n and pd.to_numeric(text, errors="coerce").notna().mean() >= PARSE_FRAC:
        return ColumnType.NUMERIC, "mostly numbers stored as text (others may be missing codes)"
    if n and _parses_as_dates(text):
        return ColumnType.DATETIME, "values parse as dates"
    if n >= MIN_ROWS_FOR_UNIQUENESS and n_unique == n and not text.str.contains(" ").any():
        return ColumnType.ID, "every value is a unique single token"
    if text.str.len().mean() >= TEXT_MIN_MEAN_LENGTH and unique_frac >= TEXT_MIN_UNIQUE_FRAC:
        return ColumnType.TEXT, "long, mostly unique strings (free text)"
    return ColumnType.CATEGORICAL, f"{n_unique} distinct text values"


def _is_row_counter(present: pd.Series) -> bool:
    """Integers like 0, 1, 2, ... or 1, 2, 3, ...: a saved row index."""
    if len(present) < MIN_ROWS_FOR_UNIQUENESS or not pd.api.types.is_integer_dtype(present):
        return False
    return bool((present.sort_values().diff().dropna() == 1).all())


def _parses_as_dates(text: pd.Series) -> bool:
    sample = text.head(1000)  # parsing is slow; a sample is enough to decide
    # The parser fills in missing parts ("May" -> May 1st this year), so only
    # strings containing a digit may count as dates.
    parsed = pd.to_datetime(sample, errors="coerce", format="mixed")
    is_date = parsed.notna() & sample.str.contains(r"\d")
    return is_date.mean() >= PARSE_FRAC
