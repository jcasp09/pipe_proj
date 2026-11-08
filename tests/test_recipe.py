import json

import numpy as np
import pandas as pd
import pytest

from preppipe.recipe import Clip, ColumnStep, Recipe, RecipeError, default_recipe
from preppipe.schema import ColumnType, infer_schema
from preppipe.split import SplitSettings, TaskType

NUM, CAT = ColumnType.NUMERIC, ColumnType.CATEGORICAL


@pytest.fixture
def recipe():
    return Recipe(
        columns={
            "age": ColumnStep(NUM, impute="median", scale="standard"),
            "income": ColumnStep(NUM, impute="median", clip=Clip(upper_percentile=99)),
            "city": ColumnStep(CAT, impute="constant", fill_value="missing", encode="onehot"),
        },
        target="default",
        task=TaskType.CLASSIFICATION,
        drop={"Unnamed: 0": "ID-like"},
        split=SplitSettings(test_size=0.25, seed=1),
    )


def make_credit_df(seed: int) -> pd.DataFrame:
    """Same columns every time, different values per seed."""
    rng = np.random.default_rng(seed)
    n = 50
    return pd.DataFrame(
        {
            "Unnamed: 0": range(n),
            "age": rng.integers(20, 80, n),
            "income": rng.normal(5000, 1500, n),
            "city": rng.choice(["NY", "LA", "SF"], n),
            "signup": pd.date_range("2024-01-01", periods=n).astype(str),
            "default": rng.integers(0, 2, n),
        }
    )


# ---------- JSON round trip ----------


def test_json_round_trip(recipe):
    assert Recipe.from_json(recipe.to_json()) == recipe


def test_json_has_metadata_and_omits_unset_options(recipe):
    d = json.loads(recipe.to_json())

    assert d["format_version"] == 1
    assert d["recipe_id"] == recipe.recipe_id
    assert "sklearn_version" in d
    assert d["columns"]["income"] == {
        "type": "numeric",
        "impute": "median",
        "clip": {"upper_percentile": 99},
    }


# ---------- rule 4: strategies, never learned values ----------


def test_recipe_does_not_depend_on_data_values():
    """Same columns, different numbers -> identical recipe. If any learned value
    (a median, a cap) leaked into the recipe, the two JSONs would differ."""
    a = make_credit_df(seed=0)
    b = make_credit_df(seed=99)
    task = TaskType.CLASSIFICATION

    recipe_a = default_recipe(infer_schema(a), target="default", task=task)
    recipe_b = default_recipe(infer_schema(b), target="default", task=task)

    assert recipe_a.to_json() == recipe_b.to_json()


def test_learned_values_are_rejected_on_load(recipe):
    d = recipe.to_dict()
    d["columns"]["income"]["impute_value"] = 5400

    with pytest.raises(RecipeError, match="never learned values"):
        Recipe.from_dict(d)


def test_learned_clip_value_is_rejected_on_load(recipe):
    d = recipe.to_dict()
    d["columns"]["income"]["clip"] = {"upper": 4980}

    with pytest.raises(RecipeError, match="unknown key"):
        Recipe.from_dict(d)


# ---------- recipe_id ----------


def test_recipe_id_is_stable_and_tracks_changes(recipe):
    same = Recipe.from_json(recipe.to_json())
    assert same.recipe_id == recipe.recipe_id

    recipe.columns["age"].scale = "robust"
    assert recipe.recipe_id != same.recipe_id


# ---------- validation: one table of invalid column steps ----------


@pytest.mark.parametrize(
    "step, message",
    [
        pytest.param(
            ColumnStep(CAT, scale="standard"), "can't be clipped or scaled", id="scale-cat"
        ),
        pytest.param(ColumnStep(CAT, impute="mean"), "impute must be one of", id="mean-on-cat"),
        pytest.param(ColumnStep(NUM, encode="onehot"), "can't be encoded", id="encode-num"),
        pytest.param(ColumnStep(NUM, impute="mode"), "impute must be one of", id="bad-impute"),
        pytest.param(ColumnStep(NUM, scale="zscore"), "scale must be one of", id="bad-scale"),
        pytest.param(ColumnStep(NUM, impute="constant"), "needs a fill_value", id="no-fill"),
        pytest.param(ColumnStep(NUM, fill_value=0), "only used with", id="fill-no-constant"),
        pytest.param(ColumnStep(NUM, clip=Clip()), "needs a lower", id="empty-clip"),
        pytest.param(
            ColumnStep(NUM, clip=Clip(upper_percentile=100)), "between 0 and 100", id="clip-100"
        ),  # fmt: skip
        pytest.param(ColumnStep(NUM, clip=Clip(99, 1)), "must be below", id="clip-reversed"),
        pytest.param(ColumnStep(ColumnType.DATETIME), "drop the column", id="datetime"),
    ],
)
def test_invalid_column_step(step, message):
    with pytest.raises(RecipeError, match=message):
        Recipe({"x": step})


def test_error_names_the_column():
    with pytest.raises(RecipeError, match="column 'city'"):
        Recipe({"city": ColumnStep(CAT, scale="standard")})


@pytest.mark.parametrize(
    "kwargs, message",
    [
        pytest.param({"columns": {}}, "at least one", id="no-columns"),
        pytest.param(
            {"target": "age", "task": TaskType.REGRESSION}, "can't also be", id="target-is-feature"
        ),  # fmt: skip
        pytest.param({"target": "y"}, "together", id="target-without-task"),
        pytest.param({"drop": {"age": "why"}}, "both kept and dropped", id="kept-and-dropped"),
    ],
)
def test_invalid_recipe(kwargs, message):
    args = {"columns": {"age": ColumnStep(NUM)}, **kwargs}
    with pytest.raises(RecipeError, match=message):
        Recipe(**args)


def test_edits_after_creation_are_checked_on_export(recipe):
    recipe.columns["city"].scale = "standard"  # invalid, but no error yet

    with pytest.raises(RecipeError, match="column 'city'"):
        recipe.to_json()


def test_unsupported_format_version(recipe):
    d = recipe.to_dict()
    d["format_version"] = 99
    with pytest.raises(RecipeError, match="format_version"):
        Recipe.from_dict(d)


# ---------- default_recipe ----------


def test_default_recipe():
    df = make_credit_df(seed=0)

    recipe = default_recipe(infer_schema(df), target="default", task=TaskType.CLASSIFICATION)

    assert recipe.columns == {
        "age": ColumnStep(NUM, impute="median", scale="standard"),
        "income": ColumnStep(NUM, impute="median", scale="standard"),
        "city": ColumnStep(CAT, impute="most_frequent", encode="onehot"),
    }
    assert set(recipe.drop) == {"Unnamed: 0", "signup"}
    assert "not handled yet" in recipe.drop["signup"]
    assert "default" not in recipe.columns and "default" not in recipe.drop
