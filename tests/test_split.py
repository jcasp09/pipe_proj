import numpy as np
import pandas as pd
import pytest

from preppipe.split import SplitSettings, TaskType, analyze_target, split_data


@pytest.fixture
def credit_df():
    """1000 rows, ~7% positive class (like Give Me Some Credit).
    Seeded, so every test run sees exactly the same data."""
    rng = np.random.default_rng(0)
    n = 1000
    return pd.DataFrame(
        {
            "age": rng.integers(20, 80, n),
            "income": rng.normal(5000, 1500, n),
            "default": (rng.random(n) < 0.07).astype(int),
        }
    )


@pytest.fixture
def housing_df():
    """Regression target: continuous prices."""
    rng = np.random.default_rng(1)
    n = 500
    return pd.DataFrame({"rooms": rng.integers(1, 8, n), "price": rng.normal(300_000, 50_000, n)})


# ---------- analyze_target ----------


@pytest.mark.parametrize(
    "values, expected",
    [
        pytest.param([0, 1, 1, 0], TaskType.CLASSIFICATION, id="binary-ints"),
        pytest.param([0.0, 1.0, None, 1.0], TaskType.CLASSIFICATION, id="binary-floats-with-nan"),
        pytest.param(["cat", "dog", "cat"], TaskType.CLASSIFICATION, id="strings"),
        pytest.param([True, False, True], TaskType.CLASSIFICATION, id="bool"),
        pytest.param([1, 2, 3, 2, 1], TaskType.CLASSIFICATION, id="few-int-levels"),
        pytest.param([1.5, 2.7, 3.1, 0.2], TaskType.REGRESSION, id="floats"),
        pytest.param(list(range(100)), TaskType.REGRESSION, id="many-int-levels"),
    ],
)
def test_detect_task(values, expected):
    assert analyze_target(pd.Series(values, name="y")).task == expected


def test_imbalance_warning(credit_df):
    info = analyze_target(credit_df["default"])

    assert set(info.class_counts) == {0, 1}
    assert any("imbalanced" in w for w in info.warnings)


def test_balanced_target_has_no_warnings():
    info = analyze_target(pd.Series([0, 1] * 50))
    assert info.warnings == []


def test_missing_target_values_are_reported():
    info = analyze_target(pd.Series([0, 1, None, 1, 0, None]))
    assert any("2 rows have no target" in w for w in info.warnings)


def test_single_valued_target_raises():
    with pytest.raises(ValueError, match="fewer than two"):
        analyze_target(pd.Series([1, 1, 1]))


# ---------- split_data: the properties that prevent leakage ----------


def test_split_sizes(credit_df):
    result = split_data(credit_df, "default")

    assert len(result.train) == 800
    assert len(result.test) == 200


def test_train_and_test_do_not_overlap(credit_df):
    """The core guarantee: no row is in both splits, and none is lost."""
    result = split_data(credit_df, "default")

    assert result.train.index.intersection(result.test.index).empty
    assert len(result.train) + len(result.test) == len(credit_df)


def test_same_seed_gives_same_split(credit_df):
    a = split_data(credit_df, "default", SplitSettings(seed=7))
    b = split_data(credit_df, "default", SplitSettings(seed=7))
    c = split_data(credit_df, "default", SplitSettings(seed=8))

    assert a.test.index.equals(b.test.index)
    assert not a.test.index.equals(c.test.index)


def test_classification_is_stratified_by_default(credit_df):
    result = split_data(credit_df, "default")

    overall = credit_df["default"].mean()
    assert result.stratified
    assert result.train["default"].mean() == pytest.approx(overall, abs=0.005)
    assert result.test["default"].mean() == pytest.approx(overall, abs=0.005)


def test_regression_is_not_stratified(housing_df):
    result = split_data(housing_df, "price")

    assert result.task == TaskType.REGRESSION
    assert not result.stratified


def test_stratify_regression_raises(housing_df):
    with pytest.raises(ValueError, match="classification"):
        split_data(housing_df, "price", SplitSettings(stratify=True))


def test_task_override(credit_df):
    """User says 'treat this as regression': no stratification."""
    result = split_data(credit_df, "default", task=TaskType.REGRESSION)
    assert not result.stratified


def test_no_target_gives_random_split(credit_df):
    result = split_data(credit_df)

    assert result.target is None
    assert result.task is None
    assert not result.stratified
    assert len(result.test) == 200


def test_rows_with_missing_target_are_dropped(credit_df):
    credit_df.loc[:9, "default"] = None  # first 10 rows lose their label

    result = split_data(credit_df, "default")

    assert result.n_dropped_missing_target == 10
    assert result.train["default"].notna().all()
    assert result.test["default"].notna().all()
    assert len(result.train) + len(result.test) == 990


def test_rare_class_gives_clear_error():
    df = pd.DataFrame({"x": range(10), "y": ["a"] * 5 + ["b"] * 4 + ["c"]})
    with pytest.raises(ValueError, match="class 'c' has only 1 row"):
        split_data(df, "y")


def test_unknown_target_raises(credit_df):
    with pytest.raises(KeyError, match="nope"):
        split_data(credit_df, "nope")


@pytest.mark.parametrize("test_size", [0, 1, 1.5, -0.2])
def test_invalid_test_size_raises(credit_df, test_size):
    with pytest.raises(ValueError, match="test_size"):
        split_data(credit_df, "default", SplitSettings(test_size=test_size))
