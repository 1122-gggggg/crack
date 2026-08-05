"""Pure-PyTorch GNN graph assembly and minimal training smoke tests."""
from __future__ import annotations

import numpy as np
import pandas as pd
from dataclasses import replace

from topology_classifier.config import GNNConfig, TrainingConfig
from topology_classifier.training.train_gnn import GNNClassifier, build_edge_graphs, train_gnn


def _features() -> pd.DataFrame:
    rows = []
    for image_index, panel in enumerate(("panel_a", "panel_a", "panel_b", "panel_b")):
        image_id = f"image_{image_index}"
        rows.extend(
            [
                {
                    "image_id": image_id,
                    "panel_id": panel,
                    "component_id": 0,
                    "edge_id": 0,
                    "u": 0,
                    "v": 1,
                    "row": 10.0,
                    "col": 10.0,
                    "length_px": 10.0 + image_index,
                    "shape_feature": 0.0,
                    "label": "crack",
                },
                {
                    "image_id": image_id,
                    "panel_id": panel,
                    "component_id": 0,
                    "edge_id": 1,
                    "u": 1,
                    "v": 2,
                    "row": 20.0,
                    "col": 20.0,
                    "length_px": 20.0 + image_index,
                    "shape_feature": 1.0,
                    "label": "craquelure",
                },
            ]
        )
    return pd.DataFrame(rows)


def test_line_graph_connects_edges_sharing_a_skeleton_node():
    frame = _features()
    frame["y"] = frame["label"].map({"crack": 0, "craquelure": 1})
    samples = build_edge_graphs(frame, ["length_px", "shape_feature"])

    assert len(samples) == 4
    assert samples[0].node_count == 2
    assert np.array_equal(samples[0].edge_index, np.asarray([[0, 1], [1, 0]]))


def test_gnn_training_writes_checkpoint_and_report(tmp_path, config):
    small = replace(
        config,
        classes=replace(config.classes, enable_other_line=False),
        gnn=GNNConfig(hidden_dim=8, num_layers=1, dropout=0.0, epochs=1),
        training=TrainingConfig(n_splits=2, group_column="panel_id", class_weighting="balanced"),
    )
    outcome = train_gnn(small, _features(), output_dir=tmp_path, epochs=1, device="cpu")

    assert outcome.aggregate["folds"] == 2
    assert outcome.model_path == tmp_path / "gnn_model.pt"
    assert outcome.model_path.is_file()
    assert (tmp_path / "gnn_report.md").is_file()
    assert (tmp_path / "gnn_training_outcome.json").is_file()

    classifier = GNNClassifier.load(outcome.model_path, device="cpu")
    frame = _features()
    probabilities = classifier.predict_proba(frame.iloc[:2].copy())
    assert probabilities.shape == (2, 2)
    assert np.allclose(probabilities.sum(axis=1), 1.0)
