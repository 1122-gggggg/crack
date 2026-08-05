"""Grouped train/validation splitting.

Edges from the same panel are highly correlated, so every split groups by
``panel_id``. Stratification is attempted first; when the label/group structure
makes it impossible, the code falls back to plain ``GroupKFold`` and says so
loudly instead of quietly leaking panels across folds.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, GroupShuffleSplit, StratifiedGroupKFold

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SplitReport:
    """Bookkeeping about how a split was produced."""

    strategy: str
    n_splits: int
    group_column: str
    group_count: int
    warnings: Tuple[str, ...] = ()

    def as_dict(self) -> dict:
        return {
            "strategy": self.strategy,
            "n_splits": self.n_splits,
            "group_column": self.group_column,
            "group_count": self.group_count,
            "warnings": list(self.warnings),
        }


def assert_no_group_leakage(groups: Sequence[str], train_index: np.ndarray, test_index: np.ndarray) -> None:
    """Fail loudly if a group appears on both sides of a split."""
    array = np.asarray(groups)
    overlap = set(array[train_index]) & set(array[test_index])
    if overlap:
        raise AssertionError(f"group leakage across split: {sorted(overlap)[:10]}")


def grouped_folds(
    labels: Sequence[int],
    groups: Sequence[str],
    n_splits: int,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], SplitReport]:
    """Grouped cross-validation folds, stratified when the data allows it.

    Raises:
        ValueError: If fewer than two groups exist -- grouped CV is meaningless
            and would silently degrade into random splitting.
    """
    label_array = np.asarray(labels)
    group_array = np.asarray(groups)
    unique_groups = np.unique(group_array)
    warnings: List[str] = []

    if unique_groups.size < 2:
        raise ValueError(
            f"grouped CV needs at least 2 groups, found {unique_groups.size}: "
            "provide panel_mapping_csv or annotate more panels"
        )

    effective_splits = int(min(n_splits, unique_groups.size))
    if effective_splits < n_splits:
        warnings.append(f"n_splits reduced from {n_splits} to {effective_splits} (only {unique_groups.size} groups)")

    folds: List[Tuple[np.ndarray, np.ndarray]] = []
    strategy = "StratifiedGroupKFold"
    try:
        splitter = StratifiedGroupKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
        folds = [(tr, te) for tr, te in splitter.split(np.zeros(len(label_array)), label_array, group_array)]
        empty = [i for i, (_, te) in enumerate(folds) if len(np.unique(label_array[te])) < 2]
        if empty:
            warnings.append(f"fold(s) {empty} contain a single class in validation")
    except ValueError as error:
        strategy = "GroupKFold"
        warnings.append(f"StratifiedGroupKFold unavailable ({error}); falling back to GroupKFold")
        logger.warning("StratifiedGroupKFold failed (%s); using GroupKFold", error)
        splitter = GroupKFold(n_splits=effective_splits)
        folds = [(tr, te) for tr, te in splitter.split(np.zeros(len(label_array)), label_array, group_array)]

    for train_index, test_index in folds:
        assert_no_group_leakage(group_array, train_index, test_index)

    report = SplitReport(
        strategy=strategy,
        n_splits=effective_splits,
        group_column="",
        group_count=int(unique_groups.size),
        warnings=tuple(warnings),
    )
    for message in warnings:
        logger.warning("split: %s", message)
    return folds, report


def grouped_holdout(
    groups: Sequence[str],
    test_size: float,
    seed: int = 42,
) -> Tuple[np.ndarray, np.ndarray]:
    """Single grouped train/test split."""
    group_array = np.asarray(groups)
    if np.unique(group_array).size < 2:
        raise ValueError("grouped holdout needs at least 2 groups")
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    train_index, test_index = next(splitter.split(np.zeros(len(group_array)), groups=group_array))
    assert_no_group_leakage(group_array, train_index, test_index)
    return train_index, test_index


def split_frame(
    frame: pd.DataFrame,
    label_column: str,
    group_column: str,
    n_splits: int,
    seed: int = 42,
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], SplitReport]:
    """Convenience wrapper that reads labels and groups from a DataFrame."""
    if label_column not in frame.columns:
        raise KeyError(f"label column {label_column!r} missing from feature frame")
    if group_column not in frame.columns:
        raise KeyError(f"group column {group_column!r} missing from feature frame")
    folds, report = grouped_folds(
        frame[label_column].to_numpy(), frame[group_column].astype(str).to_numpy(), n_splits, seed
    )
    return folds, SplitReport(
        strategy=report.strategy,
        n_splits=report.n_splits,
        group_column=group_column,
        group_count=report.group_count,
        warnings=report.warnings,
    )


def iter_folds(
    frame: pd.DataFrame, folds: Sequence[Tuple[np.ndarray, np.ndarray]]
) -> Iterator[Tuple[int, pd.DataFrame, pd.DataFrame]]:
    for index, (train_index, test_index) in enumerate(folds):
        yield index, frame.iloc[train_index], frame.iloc[test_index]


def describe_groups(frame: pd.DataFrame, group_column: str, label_column: Optional[str] = None) -> pd.DataFrame:
    """Per-group row and class counts, printed before training as a sanity check."""
    if label_column and label_column in frame.columns:
        table = frame.groupby([group_column, label_column]).size().unstack(fill_value=0)
    else:
        table = frame.groupby(group_column).size().to_frame("rows")
    return table
