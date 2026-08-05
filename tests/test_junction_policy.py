"""Junction conflict handling must be conservative and graph-aware."""
from __future__ import annotations

import pandas as pd
import pytest

from topology_classifier.inference.classify_edges import _apply_junction_policy
from topology_classifier.labels.label_types import UNCERTAIN


def _frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "image_id": ["img", "img", "img", "isolated"],
            "panel_id": ["p", "p", "p", "p2"],
            "edge_id": [0, 1, 2, 0],
            "u": [0, 1, 1, 10],
            "v": [1, 2, 3, 11],
            "raw_label": ["crack", "craquelure", "craquelure", "crack"],
            "predicted_label": ["crack", "craquelure", "craquelure", "crack"],
        }
    )


def test_outlier_at_junction_becomes_uncertain_but_supported_edges_remain():
    result = _apply_junction_policy(_frame(), "uncertain")

    image = result[result.image_id == "img"]
    assert image.loc[image.edge_id == 0, "predicted_label"].item() == UNCERTAIN
    assert image.loc[image.edge_id == 1, "predicted_label"].item() == "craquelure"
    assert image.loc[image.edge_id == 2, "predicted_label"].item() == "craquelure"
    assert result.loc[result.image_id == "isolated", "predicted_label"].item() == "crack"


def test_keep_policy_and_invalid_policy():
    original = _frame()
    kept = _apply_junction_policy(original, "keep")
    assert kept["predicted_label"].tolist() == original["predicted_label"].tolist()

    with pytest.raises(ValueError, match="junction_conflict_policy"):
        _apply_junction_policy(original, "invalid")


def test_missing_graph_columns_are_left_untouched():
    frame = _frame().drop(columns=["u", "v"])
    result = _apply_junction_policy(frame, "uncertain")
    assert result["predicted_label"].tolist() == frame["predicted_label"].tolist()
