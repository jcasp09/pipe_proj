"""Recipe data model and JSON (de)serialization.

A recipe says *how* to prepare each column ("impute with the median", "cap at the
99th percentile"), never the learned values themselves (CLAUDE.md rule 4). The
values are learned later, when the compiled pipeline is fit on the training split.
That's why one recipe can give both an unfitted pipeline (for CV) and a fitted one.
"""

import hashlib
import json
from dataclasses import asdict, dataclass, field

import sklearn

from preppipe.schema import ColumnInfo, ColumnType, suggest_exclusions
from preppipe.split import SplitSettings, TaskType

FORMAT_VERSION = 1  # bump when the JSON layout changes

NUMERIC_IMPUTE = {"mean", "median", "most_frequent", "constant"}
CATEGORICAL_IMPUTE = {"most_frequent", "constant"}
SCALERS = {"standard", "minmax", "robust"}
ENCODERS = {"onehot", "ordinal"}
RECIPE_COLUMN_TYPES = {ColumnType.NUMERIC, ColumnType.CATEGORICAL}


class RecipeError(ValueError):
    """An invalid recipe: the message says which column and what's wrong."""


@dataclass
class Clip:
    """Cap values at training-split percentiles (0-100). Rows are never dropped."""

    lower_percentile: float | None = None
    upper_percentile: float | None = None


@dataclass
class ColumnStep:
    """What to do with one column, applied in order: impute -> clip -> scale/encode."""

    type: ColumnType
    impute: str | None = None
    fill_value: float | str | None = None  # a user choice, only with impute="constant"
    clip: Clip | None = None  # numeric only
    scale: str | None = None  # numeric only
    encode: str | None = None  # categorical only

    def validate(self, name: str) -> None:
        def fail(msg: str):
            raise RecipeError(f"column {name!r}: {msg}")

        if self.type not in RECIPE_COLUMN_TYPES:
            fail(f"type {self.type!r} can't be prepared yet; drop the column instead")

        allowed_impute = NUMERIC_IMPUTE if self.type == ColumnType.NUMERIC else CATEGORICAL_IMPUTE
        if self.impute is not None and self.impute not in allowed_impute:
            fail(f"impute must be one of {sorted(allowed_impute)}, got {self.impute!r}")
        if self.impute == "constant" and self.fill_value is None:
            fail("impute='constant' needs a fill_value")
        if self.fill_value is not None and self.impute != "constant":
            fail("fill_value is only used with impute='constant'")

        if self.type == ColumnType.NUMERIC:
            if self.encode is not None:
                fail("numeric columns can't be encoded")
            if self.scale is not None and self.scale not in SCALERS:
                fail(f"scale must be one of {sorted(SCALERS)}, got {self.scale!r}")
            if self.clip is not None:
                _validate_clip(self.clip, fail)
        else:
            if self.clip is not None or self.scale is not None:
                fail("categorical columns can't be clipped or scaled")
            if self.encode is not None and self.encode not in ENCODERS:
                fail(f"encode must be one of {sorted(ENCODERS)}, got {self.encode!r}")


@dataclass
class Recipe:
    columns: dict[str, ColumnStep]
    target: str | None = None
    task: TaskType | None = None
    drop: dict[str, str] = field(default_factory=dict)  # column -> reason
    split: SplitSettings = field(default_factory=SplitSettings)

    def __post_init__(self):
        self.validate()

    def validate(self) -> None:
        """Runs on creation and again on export, so edits made after creation are checked."""
        if not self.columns:
            raise RecipeError("a recipe needs at least one feature column")
        if (self.target is None) != (self.task is None):
            raise RecipeError("target and task must be given together")
        if self.target is not None and (self.target in self.columns or self.target in self.drop):
            raise RecipeError(f"target {self.target!r} can't also be a feature or dropped")
        both = set(self.columns) & set(self.drop)
        if both:
            raise RecipeError(f"columns both kept and dropped: {sorted(both)}")
        for name, step in self.columns.items():
            step.validate(name)

    @property
    def recipe_id(self) -> str:
        """Short hash of the content: same strategies -> same ID, any change -> new ID."""
        canonical = json.dumps(self._content(), sort_keys=True)
        return hashlib.sha256(canonical.encode()).hexdigest()[:12]

    def to_dict(self) -> dict:
        self.validate()
        return {
            "format_version": FORMAT_VERSION,
            "recipe_id": self.recipe_id,
            "sklearn_version": sklearn.__version__,
            **self._content(),
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)

    @classmethod
    def from_dict(cls, d: dict) -> "Recipe":
        _check_keys(d, RECIPE_KEYS, "recipe")
        if d.get("format_version") != FORMAT_VERSION:
            raise RecipeError(f"unsupported recipe format_version {d.get('format_version')!r}")
        _check_keys(d.get("split", {}), SPLIT_KEYS, "split")
        return cls(
            columns={name: _step_from_dict(name, s) for name, s in d["columns"].items()},
            target=d.get("target"),
            task=TaskType(d["task"]) if d.get("task") else None,
            drop=dict(d.get("drop", {})),
            split=SplitSettings(**d.get("split", {})),
        )

    @classmethod
    def from_json(cls, text: str) -> "Recipe":
        return cls.from_dict(json.loads(text))

    def _content(self) -> dict:
        """Everything that defines the recipe (no metadata like versions or the ID)."""
        return {
            "target": self.target,
            "task": self.task,
            "split": asdict(self.split),
            "drop": self.drop,
            "columns": {name: _step_to_dict(step) for name, step in self.columns.items()},
        }


def default_recipe(
    schema: dict[str, ColumnInfo],
    target: str | None = None,
    task: TaskType | None = None,
    split: SplitSettings | None = None,
) -> Recipe:
    """Smart defaults: impute + scale numerics, impute + one-hot encode categoricals.

    Imputation is on even for columns without missing values: data sent to the
    API later may have gaps that the training split didn't.
    """
    drop = {name: reason for name, reason in suggest_exclusions(schema).items() if name != target}
    columns = {}
    for name, info in schema.items():
        if name == target or name in drop:
            continue
        if info.type == ColumnType.NUMERIC:
            columns[name] = ColumnStep(ColumnType.NUMERIC, impute="median", scale="standard")
        elif info.type == ColumnType.CATEGORICAL:
            columns[name] = ColumnStep(
                ColumnType.CATEGORICAL, impute="most_frequent", encode="onehot"
            )
        else:
            drop[name] = f"{info.type} columns are not handled yet"
    return Recipe(columns, target=target, task=task, drop=drop, split=split or SplitSettings())


# ---------- (de)serialization helpers ----------

RECIPE_KEYS = {"format_version", "recipe_id", "sklearn_version", "target", "task", "split",
               "drop", "columns"}  # fmt: skip
SPLIT_KEYS = {"test_size", "seed", "stratify"}
STEP_KEYS = {"type", "impute", "fill_value", "clip", "scale", "encode"}
CLIP_KEYS = {"lower_percentile", "upper_percentile"}


def _step_to_dict(step: ColumnStep) -> dict:
    """Leave out unset options so the JSON stays short and readable."""
    d = {key: value for key, value in asdict(step).items() if value is not None}
    if "clip" in d:
        d["clip"] = {key: value for key, value in d["clip"].items() if value is not None}
    return d


def _step_from_dict(name: str, d: dict) -> ColumnStep:
    _check_keys(d, STEP_KEYS, f"column {name!r}")
    clip = d.get("clip")
    if clip is not None:
        _check_keys(clip, CLIP_KEYS, f"column {name!r} clip")
        clip = Clip(**clip)
    return ColumnStep(**{**d, "type": ColumnType(d["type"]), "clip": clip})


def _check_keys(d: dict, allowed: set[str], where: str) -> None:
    unknown = set(d) - allowed
    if unknown:
        raise RecipeError(
            f"unknown key(s) {sorted(unknown)} in {where}. A recipe stores strategies "
            "(e.g. 'impute': 'median'), never learned values."
        )


def _validate_clip(clip: Clip, fail) -> None:
    low, high = clip.lower_percentile, clip.upper_percentile
    if low is None and high is None:
        fail("clip needs a lower_percentile, an upper_percentile, or both")
    for p in (low, high):
        if p is not None and not 0 < p < 100:
            fail(f"clip percentiles must be between 0 and 100, got {p}")
    if low is not None and high is not None and low >= high:
        fail(f"clip lower_percentile ({low}) must be below upper_percentile ({high})")
