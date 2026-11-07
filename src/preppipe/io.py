"""Loading CSV files, with delimiter and header-row detection.

Detection is a best guess: the result reports what was detected, and the caller
(UI) shows it to the user, who can override it by passing `delimiter` / `has_header` / `decimal`.
"""

import csv
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

CANDIDATE_DELIMITERS = [",", ";", "\t", "|"]
SAMPLE_BYTES = 64_000  # how much of the file we inspect for detection
SAMPLE_ROWS = 50
MISSING_TOKENS = {"", "na", "n/a", "nan", "null", "none", "?"}
DECIMAL_COMMA = re.compile(r"^-?\d+,\d+$")  # e.g. "2,6" meaning 2.6


@dataclass
class CsvLoadResult:
    df: pd.DataFrame
    delimiter: str
    has_header: bool
    decimal: str


def load_csv(
    path: str | Path,
    *,
    delimiter: str | None = None,
    has_header: bool | None = None,
    decimal: str | None = None,
    encoding: str = "utf-8",
) -> CsvLoadResult:
    """Load a CSV file, detecting the delimiter and header row unless given.

    Without a header row, columns are named col_0, col_1, ... so they can
    still be referenced by name.
    """
    path = Path(path)
    with path.open(encoding=encoding, newline="") as f:
        sample = f.read(SAMPLE_BYTES)
    if not sample.strip():
        raise ValueError(f"{path.name} is empty")

    if delimiter is None:
        delimiter = detect_delimiter(sample)
    if has_header is None:
        has_header = detect_header(sample, delimiter)
    if decimal is None:
        decimal = detect_decimal(sample, delimiter)

    df = pd.read_csv(
        path,
        sep=delimiter,
        header=0 if has_header else None,
        decimal=decimal,
        encoding=encoding,
    )
    if not has_header:
        df.columns = [f"col_{i}" for i in range(df.shape[1])]
    return CsvLoadResult(df=df, delimiter=delimiter, has_header=has_header, decimal=decimal)


def detect_delimiter(sample: str) -> str:
    """Guess the delimiter from a text sample; fall back to a comma."""
    try:
        return csv.Sniffer().sniff(sample, delimiters="".join(CANDIDATE_DELIMITERS)).delimiter
    except csv.Error:
        return ","


def detect_decimal(sample: str, delimiter: str) -> str:
    """Return "," if numbers use decimal commas (common in European CSVs), else ".".

    Only possible when the delimiter is not itself a comma.
    """
    if delimiter == ",":
        return "."
    cells = (cell.strip() for row in _parse_rows(sample, delimiter) for cell in row)
    return "," if any(DECIMAL_COMMA.match(cell) for cell in cells) else "."


def detect_header(sample: str, delimiter: str) -> bool:
    """Guess whether the first row is a header.

    Rule: a header row contains no numbers, and at least one column has a
    text value in the first row but mostly numbers below it. If no column is
    numeric we can't tell that way, so we fall back to csv.Sniffer.
    """
    rows = _parse_rows(sample, delimiter)
    if len(rows) < 2:
        return False
    first, body = rows[0], rows[1:]

    if any(_is_number(cell) for cell in first):
        return False

    for col in range(len(first)):
        values = [row[col] for row in body if col < len(row)]
        if first[col].strip() and _mostly_numeric(values):
            return True

    try:
        return csv.Sniffer().has_header(sample)
    except csv.Error:
        return False


def _parse_rows(sample: str, delimiter: str) -> list[list[str]]:
    lines = sample.splitlines()
    if len(sample) >= SAMPLE_BYTES:
        lines = lines[:-1]  # last line may be cut off mid-row
    reader = csv.reader(lines[:SAMPLE_ROWS], delimiter=delimiter)
    return [row for row in reader if row]


def _is_number(cell: str) -> bool:
    cell = cell.strip()
    if DECIMAL_COMMA.match(cell):
        cell = cell.replace(",", ".")
    try:
        float(cell)
    except ValueError:
        return False
    return True


def _mostly_numeric(values: list[str], threshold: float = 0.8) -> bool:
    """True if at least `threshold` of the non-missing values parse as numbers."""
    present = [v for v in values if v.strip().lower() not in MISSING_TOKENS]
    if not present:
        return False
    return sum(_is_number(v) for v in present) / len(present) >= threshold
