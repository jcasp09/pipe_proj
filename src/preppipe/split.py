"""Target analysis and the train/test split.

The split happens once, before any EDA or fitting. After it, the test set is
only ever transformed (CLAUDE.md rules 1-2).
"""

from dataclasses import dataclass, field
from enum import StrEnum

import pandas as pd
from sklearn.model_selection import train_test_split


class TaskType(StrEnum):
    CLASSIFICATION = "classification"
    REGRESSION = "regression"


MAX_CLASSES = 20  # integer targets with more distinct values than this are regression
IMBALANCE_WARN_FRAC = 0.10  # warn when the smallest class is rarer than this


@dataclass
class TargetInfo:
    task: TaskType
    reason: str
    class_counts: dict | None = None  # classification only
    warnings: list[str] = field(default_factory=list)


@dataclass
class SplitSettings:
    test_size: float = 0.2
    seed: int = 42
    stratify: bool | None = None  # None = stratify for classification, not regression


@dataclass
class SplitResult:
    train: pd.DataFrame
    test: pd.DataFrame
    settings: SplitSettings
    target: str | None
    task: TaskType | None
    stratified: bool
    n_dropped_missing_target: int


def analyze_target(y: pd.Series) -> TargetInfo:
    """Detect the task type and summarize the target. Counts only; nothing is fit."""
    present = y.dropna()
    if present.nunique() < 2:
        raise ValueError(f"target {y.name!r} has fewer than two distinct values")

    task, reason = _detect_task(present)
    info = TargetInfo(task=task, reason=reason)

    n_missing = int(y.isna().sum())
    if n_missing:
        info.warnings.append(f"{n_missing} rows have no target value and will be dropped")

    if task == TaskType.CLASSIFICATION:
        counts = present.value_counts()
        info.class_counts = counts.to_dict()
        minority_frac = counts.min() / counts.sum()
        if minority_frac < IMBALANCE_WARN_FRAC:
            info.warnings.append(
                f"imbalanced classes: '{counts.idxmin()}' is only {minority_frac:.1%} of rows. "
                "Use stratified CV and metrics like PR-AUC; if you resample, do it inside "
                "training folds only."
            )
    return info


def split_data(
    df: pd.DataFrame,
    target: str | None = None,
    settings: SplitSettings | None = None,
    task: TaskType | None = None,
) -> SplitResult:
    """Split once into train and test.

    Rows with a missing target are dropped first. `task` overrides detection
    (the user confirms it in the UI). Without a target, the split is random.
    """
    settings = settings or SplitSettings()
    if not 0 < settings.test_size < 1:
        raise ValueError(f"test_size must be between 0 and 1, got {settings.test_size}")

    n_dropped = 0
    stratify_labels = None
    if target is None:
        if settings.stratify:
            raise ValueError("a stratified split needs a target column")
        stratified = False
    else:
        if target not in df.columns:
            raise KeyError(f"target column {target!r} not found")
        has_target = df[target].notna()
        n_dropped = int((~has_target).sum())
        df = df[has_target]

        task = task or analyze_target(df[target]).task
        stratified = (
            settings.stratify if settings.stratify is not None else task == TaskType.CLASSIFICATION
        )
        if stratified:
            if task == TaskType.REGRESSION:
                raise ValueError("stratified split needs a classification target")
            _check_stratifiable(df[target])
            stratify_labels = df[target]

    train, test = train_test_split(
        df, test_size=settings.test_size, random_state=settings.seed, stratify=stratify_labels
    )
    return SplitResult(
        train=train,
        test=test,
        settings=settings,
        target=target,
        task=task if target is not None else None,
        stratified=stratified,
        n_dropped_missing_target=n_dropped,
    )


def _detect_task(present: pd.Series) -> tuple[TaskType, str]:
    if pd.api.types.is_bool_dtype(present) or not pd.api.types.is_numeric_dtype(present):
        return TaskType.CLASSIFICATION, "non-numeric labels"
    n_unique = present.nunique()
    is_whole = bool((present == present.round()).all())  # 0.0/1.0 counts as whole
    if is_whole and n_unique <= MAX_CLASSES:
        return TaskType.CLASSIFICATION, f"{n_unique} distinct whole-number values"
    return TaskType.REGRESSION, f"{n_unique} distinct numeric values"


def _check_stratifiable(y: pd.Series) -> None:
    """Stratifying needs every class in both splits, so at least 2 rows per class."""
    counts = y.value_counts()
    rare = counts[counts < 2]
    if not rare.empty:
        raise ValueError(
            f"class '{rare.index[0]}' has only {rare.iloc[0]} row; a stratified split needs "
            "at least 2 per class. Merge rare classes or turn off stratify."
        )
