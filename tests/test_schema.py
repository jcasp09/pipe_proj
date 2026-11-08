import pandas as pd
import pytest

from preppipe.io import load_csv
from preppipe.schema import ColumnType, infer_column, infer_schema, suggest_exclusions


def col(values, name="x"):
    """Shorthand for building a named Series in test cases."""
    return pd.Series(values, name=name)


# ---------- infer_column: one table of (column, expected type) cases ----------


@pytest.mark.parametrize(
    "series, expected",
    [
        pytest.param(col([1.5, 2.0, None, 3.2]), ColumnType.NUMERIC, id="floats"),
        pytest.param(
            col([25, 26, 27, 28], name="age"), ColumnType.NUMERIC, id="few-consecutive-ints"
        ),
        pytest.param(col(list(range(100)), name="paid"), ColumnType.ID, id="row-counter"),
        pytest.param(col([True, False, True]), ColumnType.CATEGORICAL, id="bool"),
        pytest.param(col(["red", "blue", "red", None]), ColumnType.CATEGORICAL, id="strings"),
        pytest.param(
            col(["5", "7", "?", "9", "3", "4", "8", "2", "1", "6"]),
            ColumnType.NUMERIC,
            id="numbers-with-question-mark",
        ),
        pytest.param(
            col(["2024-01-05", "2024-02-11", "2024-03-02"]), ColumnType.DATETIME, id="date-strings"
        ),
        pytest.param(col(["May", "June", "March"]), ColumnType.CATEGORICAL, id="month-names"),
        pytest.param(
            col(pd.to_datetime(["2024-01-05", "2024-02-11"])),
            ColumnType.DATETIME,
            id="datetime-dtype",
        ),
        pytest.param(col([3, 1, 2], name="Unnamed: 0"), ColumnType.ID, id="saved-index-name"),
        pytest.param(
            col([10, 11, 11, 12, 13], name="patient_nbr"), ColumnType.ID, id="id-name-with-repeats"
        ),
        pytest.param(col([f"u{i:04d}x" for i in range(50)]), ColumnType.ID, id="unique-codes"),
        pytest.param(
            col([f"Customer review number {i}, quite long" for i in range(50)]),
            ColumnType.TEXT,
            id="free-text",
        ),
    ],
)
def test_infer_column_type(series, expected):
    info = infer_column(series)
    assert info.type == expected, info.reason  # on failure, pytest also prints the reason


def test_name_containing_id_is_not_id():
    """'paid' ends in 'id' but is not an ID column: the name rule needs '_id' or 'id'."""
    assert infer_column(col([0, 1, 0, 1], name="paid")).type == ColumnType.NUMERIC


def test_few_unique_strings_are_not_id():
    """With few rows, 'all unique' is weak evidence: stay categorical."""
    assert infer_column(col(["a", "b", "c"])).type == ColumnType.CATEGORICAL


def test_infer_column_stats():
    info = infer_column(col([1.0, None, 1.0, None], name="income"))

    assert info.name == "income"
    assert info.n_unique == 1
    assert info.missing_frac == 0.5
    assert info.reason  # always explains itself


# ---------- infer_schema + suggest_exclusions on one small messy table ----------


@pytest.fixture
def messy_df():
    """A table with one column per kind of problem. Local to this file:
    only these tests use it, so it doesn't belong in conftest.py."""
    n = 40
    return pd.DataFrame(
        {
            "Unnamed: 0": range(n),  # saved index
            "age": [20 + i % 30 for i in range(n)],  # fine
            "country": ["US"] * n,  # constant
            "notes": [None] * (n - 1) + ["x"],  # near-empty
            "target": [i % 2 for i in range(n)],  # fine
        }
    )


def test_infer_schema_keeps_column_order(messy_df):
    schema = infer_schema(messy_df)
    assert list(schema) == list(messy_df.columns)


def test_suggest_exclusions(messy_df):
    schema = infer_schema(messy_df)

    excluded = suggest_exclusions(messy_df, schema)

    assert set(excluded) == {"Unnamed: 0", "country", "notes"}
    assert "ID" in excluded["Unnamed: 0"]
    assert "constant" in excluded["country"]
    assert "near-empty" in excluded["notes"]


def test_all_missing_column_is_near_empty_not_id():
    """UCI Air Quality's trailing ';;' columns load as 'Unnamed: 15', all NaN."""
    df = pd.DataFrame({"CO": [2.6, 2.0], "Unnamed: 15": [None, None]})

    excluded = suggest_exclusions(df, infer_schema(df))

    assert excluded == {"Unnamed: 15": "near-empty (100% missing)"}


# ---------- integration: io + schema together on a file ----------


def test_load_then_infer_credit_like_file(write_csv):
    """Give Me Some Credit shape: unnamed saved index, binary target, missing income.
    `write_csv` comes from conftest.py, shared with test_io.py."""
    rows = [f"{i},{i % 2},{30 + i % 7},{'NA' if i % 5 == 0 else 4000 + i}" for i in range(1, 41)]
    path = write_csv(",SeriousDlqin2yrs,age,MonthlyIncome\n" + "\n".join(rows) + "\n")

    df = load_csv(path).df
    schema = infer_schema(df)

    assert {name: info.type for name, info in schema.items()} == {
        "Unnamed: 0": ColumnType.ID,
        "SeriousDlqin2yrs": ColumnType.NUMERIC,
        "age": ColumnType.NUMERIC,
        "MonthlyIncome": ColumnType.NUMERIC,
    }
    assert schema["MonthlyIncome"].missing_frac == pytest.approx(0.2)
    assert suggest_exclusions(df, schema) == {"Unnamed: 0": "ID-like (no predictive meaning)"}
