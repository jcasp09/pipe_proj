import pytest

from preppipe.io import detect_delimiter, detect_header, load_csv

# ---------- detect_delimiter: small pure functions, tested directly on strings ----------


@pytest.mark.parametrize("delim", [",", ";", "\t", "|"])
def test_detect_delimiter(delim):
    sample = delim.join(["a", "b", "c"]) + "\n" + delim.join(["1", "2", "3"]) + "\n"
    assert detect_delimiter(sample) == delim


def test_detect_delimiter_ignores_commas_inside_quotes():
    sample = 'name;city\n"Smith, J";"Paris, FR"\n"Doe, A";"Rome, IT"\n'
    assert detect_delimiter(sample) == ";"


def test_detect_delimiter_falls_back_to_comma():
    assert detect_delimiter("justonecolumn\nvalue\n") == ","


# ---------- detect_header ----------


@pytest.mark.parametrize(
    "sample, expected",
    [
        pytest.param("age,income\n35,5000\n41,6200\n", True, id="text-over-numbers"),
        pytest.param("35,5000\n41,6200\n29,4100\n", False, id="all-numbers"),
        pytest.param("age,income\n35,NA\n41,\n29,4100\n", True, id="missing-values-in-body"),
        pytest.param(",target,age\n1,0,35\n2,1,41\n", True, id="unnamed-index-column"),
        pytest.param("2004,5000\n2005,6200\n", False, id="numeric-first-row"),
    ],
)
def test_detect_header(sample, expected):
    assert detect_header(sample, ",") is expected


def test_detect_header_single_row_is_not_header():
    assert detect_header("a,b,c\n", ",") is False


# ---------- load_csv: end to end on real files (via the write_csv fixture) ----------


def test_load_csv_with_header(write_csv):
    path = write_csv("age,income,target\n35,5000,0\n41,6200,1\n")

    result = load_csv(path)

    assert result.has_header
    assert result.delimiter == ","
    assert list(result.df.columns) == ["age", "income", "target"]
    assert result.df.shape == (2, 3)


def test_load_csv_without_header_names_columns(write_csv):
    path = write_csv("35,5000,0\n41,6200,1\n")

    result = load_csv(path)

    assert not result.has_header
    assert list(result.df.columns) == ["col_0", "col_1", "col_2"]
    assert len(result.df) == 2  # first row kept as data, not eaten as header


def test_load_csv_semicolon(write_csv):
    path = write_csv("Date;Time;CO\n10/03/2004;18.00.00;2.6\n10/03/2004;19.00.00;2.0\n")

    result = load_csv(path)

    assert result.delimiter == ";"
    assert list(result.df.columns) == ["Date", "Time", "CO"]


def test_load_csv_decimal_comma(write_csv):
    """UCI Air Quality style: ';' delimiter and '2,6' meaning 2.6."""
    path = write_csv("Date;CO;T\n10/03/2004;2,6;13,6\n10/03/2004;2;-200\n")

    result = load_csv(path)

    assert result.decimal == ","
    assert result.df["CO"].tolist() == [2.6, 2.0]
    assert result.df["T"].tolist() == [13.6, -200.0]


def test_load_csv_overrides_skip_detection(write_csv):
    """User corrections win over detection (header confirmation step)."""
    path = write_csv("a,b\n1,2\n")

    result = load_csv(path, has_header=False)

    assert not result.has_header
    assert result.df.shape == (2, 2)
    assert result.df.iloc[0].tolist() == ["a", "b"]


def test_load_csv_empty_file_raises(write_csv):
    path = write_csv("")
    with pytest.raises(ValueError, match="empty"):
        load_csv(path)


def test_load_csv_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        load_csv(tmp_path / "nope.csv")
