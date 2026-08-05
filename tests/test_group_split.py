"""Grouped splitting must never place one panel on both sides of a fold."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from topology_classifier.training.splits import (
    assert_no_group_leakage,
    describe_groups,
    grouped_folds,
    grouped_holdout,
    split_frame,
)


def _synthetic(panels: int = 6, per_panel: int = 20, seed: int = 0):
    rng = np.random.default_rng(seed)
    groups = np.repeat([f"panel_{i:02d}" for i in range(panels)], per_panel)
    labels = rng.integers(0, 2, size=groups.size)
    # Guarantee both classes exist in every panel so stratification is feasible.
    for index in range(panels):
        start = index * per_panel
        labels[start] = 0
        labels[start + 1] = 1
    return labels, groups


def test_folds_never_share_a_group():
    labels, groups = _synthetic()

    folds, report = grouped_folds(labels, groups, n_splits=3, seed=42)

    assert len(folds) == 3
    assert report.group_count == 6
    for train_index, test_index in folds:
        assert set(groups[train_index]).isdisjoint(set(groups[test_index]))
        assert train_index.size + test_index.size == labels.size


def test_every_sample_is_tested_exactly_once():
    labels, groups = _synthetic()

    folds, _ = grouped_folds(labels, groups, n_splits=3, seed=42)

    tested = np.concatenate([test for _, test in folds])
    assert np.array_equal(np.sort(tested), np.arange(labels.size))


def test_single_group_is_rejected_instead_of_leaking():
    labels = np.array([0, 1, 0, 1])
    groups = np.array(["panel_a"] * 4)

    with pytest.raises(ValueError, match="at least 2 groups"):
        grouped_folds(labels, groups, n_splits=3)


def test_n_splits_is_reduced_and_reported_when_groups_are_few():
    labels, groups = _synthetic(panels=2, per_panel=10)

    folds, report = grouped_folds(labels, groups, n_splits=5, seed=42)

    assert len(folds) == 2
    assert report.n_splits == 2
    assert any("reduced" in warning for warning in report.warnings)


def test_leakage_assertion_fires():
    groups = np.array(["a", "a", "b", "b"])

    with pytest.raises(AssertionError, match="group leakage"):
        assert_no_group_leakage(groups, np.array([0, 1]), np.array([1, 2]))


def test_grouped_holdout_keeps_panels_whole():
    _, groups = _synthetic()

    train_index, test_index = grouped_holdout(groups, test_size=0.34, seed=7)

    assert set(groups[train_index]).isdisjoint(set(groups[test_index]))
    assert test_index.size > 0


def test_split_frame_uses_the_named_columns():
    labels, groups = _synthetic()
    frame = pd.DataFrame({"y": labels, "panel_id": groups, "length_px": 1.0})

    folds, report = split_frame(frame, label_column="y", group_column="panel_id", n_splits=3, seed=1)

    assert report.group_column == "panel_id"
    assert report.strategy in {"StratifiedGroupKFold", "GroupKFold"}
    for train_index, test_index in folds:
        assert set(frame["panel_id"].iloc[train_index]).isdisjoint(frame["panel_id"].iloc[test_index])


def test_split_frame_rejects_a_missing_group_column():
    frame = pd.DataFrame({"y": [0, 1, 0, 1]})

    with pytest.raises(KeyError, match="panel_id"):
        split_frame(frame, label_column="y", group_column="panel_id", n_splits=2)


def test_describe_groups_reports_per_panel_class_counts():
    labels, groups = _synthetic(panels=3, per_panel=10)
    frame = pd.DataFrame({"label": labels, "panel_id": groups})

    table = describe_groups(frame, "panel_id", "label")

    assert table.shape[0] == 3
    assert int(table.to_numpy().sum()) == labels.size
