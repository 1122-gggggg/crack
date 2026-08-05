"""End-to-end smoke test: synthetic Stage-1 outputs through every CLI command.

The dataset is deliberately tiny and synthetic. It exists to prove the commands
run and produce the artefacts they claim to -- not to demonstrate accuracy.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pytest
import yaml
from conftest import draw_line

from topology_classifier.cli import build_parser, main

CANVAS: Tuple[int, int] = (240, 240)
PANELS: Dict[str, str] = {
    "img_a1": "panel_a",
    "img_a2": "panel_a",
    "img_b1": "panel_b",
    "img_b2": "panel_b",
}


def _crack_lines(mask: np.ndarray, thickness: int, offset: int) -> np.ndarray:
    """Two long, straight, isolated lines: the crack-like structures."""
    for col in (30 + offset, 70 + offset):
        draw_line(mask, (20, col), (220, col), thickness=thickness)
    return mask


def _craquelure_grid(mask: np.ndarray, thickness: int, offset: int) -> np.ndarray:
    """A closed-cell network: the craquelure-like structure."""
    for row in (40, 80, 120, 160):
        draw_line(mask, (row, 140 + offset), (row, 220), thickness=thickness)
    for col in (140 + offset, 180, 220):
        draw_line(mask, (40, col), (160, col), thickness=thickness)
    return mask


def _write_image(path: Path, mask: np.ndarray) -> None:
    import cv2

    image = np.full((*mask.shape, 3), 150, dtype=np.uint8)
    image[mask.astype(bool)] = (40, 45, 55)
    cv2.imwrite(str(path), image)


@pytest.fixture(scope="module")
def dataset(tmp_path_factory) -> Path:
    """Build a 4-image / 2-panel synthetic dataset and return the config path."""
    root = tmp_path_factory.mktemp("smoke")
    images = root / "images"
    prob_dir = root / "rift_prob"
    mask_dir = root / "rift_mask"
    labels_dir = root / "labels"
    for directory in (images, prob_dir, mask_dir, labels_dir):
        directory.mkdir(parents=True, exist_ok=True)

    for index, image_id in enumerate(sorted(PANELS)):
        offset = index % 2  # a one-pixel shift so the images are not identical
        mask = np.zeros(CANVAS, dtype=np.uint8)
        _crack_lines(mask, thickness=3, offset=offset)
        _craquelure_grid(mask, thickness=3, offset=offset)

        probability = np.where(mask.astype(bool), 0.92, 0.04).astype(np.float32)
        np.save(prob_dir / f"{image_id}_prob.npy", probability)
        np.save(mask_dir / f"{image_id}_mask.npy", mask)
        _write_image(images / f"{image_id}.png", mask)

        label = np.zeros(CANVAS, dtype=np.uint8)
        crack_band = _crack_lines(np.zeros(CANVAS, dtype=np.uint8), thickness=9, offset=offset)
        grid_band = _craquelure_grid(np.zeros(CANVAS, dtype=np.uint8), thickness=9, offset=offset)
        label[crack_band.astype(bool)] = 1
        label[grid_band.astype(bool)] = 2
        np.save(labels_dir / f"{image_id}.npy", label)

    mapping = root / "panel_mapping.csv"
    mapping.write_text(
        "image_id,panel_id\n" + "".join(f"{k},{v}\n" for k, v in sorted(PANELS.items())),
        encoding="utf-8",
    )

    payload = {
        "data": {
            "images_dir": images.as_posix(),
            "rift_prob_dir": prob_dir.as_posix(),
            "rift_mask_dir": mask_dir.as_posix(),
            "labels_dir": labels_dir.as_posix(),
            "panel_mapping_csv": mapping.as_posix(),
            "output_dir": (root / "outputs").as_posix(),
        },
        "classes": {"enable_other_line": False},
        "preprocessing": {"save_debug_masks": False},
        "model": {"baseline_type": "random_forest", "minimum_prediction_confidence": 0.55},
        "training": {"n_splits": 2, "n_estimators": 40, "max_depth": 5},
        "runtime": {"cache_dir": (root / "cache").as_posix(), "log_level": "WARNING"},
    }
    config_path = root / "topology.yaml"
    config_path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
    return config_path


def test_cli_exposes_every_required_subcommand():
    parser = build_parser()
    actions = [a for a in parser._subparsers._group_actions if hasattr(a, "choices")]  # noqa: SLF001
    commands = set(actions[0].choices)

    assert commands == {
        "build-graphs",
        "export-edge-review",
        "build-features",
        "train-baseline",
        "train-gnn",
        "infer",
        "evaluate",
        "visualize-graph",
    }


def test_end_to_end_pipeline(dataset):
    config_path = str(dataset)
    root = dataset.parent
    outputs = root / "outputs"

    assert main(["build-graphs", "--config", config_path]) == 0
    summary = json.loads((outputs / "graphs" / "graph_summary.json").read_text(encoding="utf-8"))
    assert len(summary) == len(PANELS)
    assert all(row["edge_count"] > 0 for row in summary)
    assert all(row["probability_available"] for row in summary)

    assert main(["export-edge-review", "--config", config_path, "--max-edges", "6", "--no-crops"]) == 0
    review = outputs / "review" / "edge_review_all.csv"
    assert review.is_file()
    header, *rows = review.read_text(encoding="utf-8-sig").strip().splitlines()
    assert "label" in header.split(",")
    assert rows, "review CSV must list edges"
    label_column = header.split(",").index("label")
    assert all(row.split(",")[label_column] == "" for row in rows), "labels must ship empty"

    assert main(["build-features", "--config", config_path]) == 0
    feature_summary = json.loads(
        (outputs / "features" / "feature_summary.json").read_text(encoding="utf-8")
    )
    assert feature_summary["rows"] > 0
    assert feature_summary["feature_count"] > 20
    assert feature_summary["label_counts"].get("crack", 0) > 0
    assert feature_summary["label_counts"].get("craquelure", 0) > 0

    assert main(["train-baseline", "--config", config_path]) == 0
    outcome = json.loads((outputs / "baseline" / "training_outcome.json").read_text(encoding="utf-8"))
    assert outcome["aggregate"]["folds"] == 2
    assert 0.0 <= outcome["aggregate"]["accuracy_mean"] <= 1.0
    assert (outputs / "baseline" / "baseline_model.joblib").is_file()
    assert (outputs / "baseline" / "baseline_report.md").is_file()

    assert main(["train-gnn", "--config", config_path, "--epochs", "1", "--device", "cpu"]) == 0
    gnn_outcome = json.loads((outputs / "gnn" / "gnn_training_outcome.json").read_text(encoding="utf-8"))
    assert gnn_outcome["aggregate"]["folds"] == 2
    assert (outputs / "gnn" / "gnn_model.pt").is_file()

    assert main(["infer", "--config", config_path]) == 0
    predictions = outputs / "predictions"
    inference = json.loads((predictions / "inference_summary.json").read_text(encoding="utf-8"))
    assert len(inference) == len(PANELS)
    for image_id in PANELS:
        assert (predictions / f"{image_id}_edge_predictions.csv").is_file()
        class_mask = np.load(predictions / f"{image_id}_class_mask.npy")
        assert class_mask.shape == CANVAS
        assert set(np.unique(class_mask)).issubset({0, 1, 2, 255})

    assert main(["evaluate", "--config", config_path]) == 0
    report = (outputs / "evaluation" / "evaluation_report.md").read_text(encoding="utf-8")
    assert "in-sample" in report
    assert (outputs / "evaluation" / "edge_evaluation.csv").is_file()

    assert main(["infer", "--config", config_path, "--model-type", "gnn"]) == 0
    assert main(["evaluate", "--config", config_path, "--model-type", "gnn"]) == 0

    assert main(["visualize-graph", "--config", config_path]) == 0
    assert (outputs / "visualization" / "img_a1_graph.png").is_file()
    assert (outputs / "visualization" / "img_a1_classes.png").is_file()

    errors = outputs / "errors.jsonl"
    assert not errors.is_file() or not errors.read_text(encoding="utf-8").strip()


def test_class_mask_only_covers_the_input_mask(dataset):
    """Rasterization redistributes foreground pixels; it never invents them."""
    root = dataset.parent
    predictions = root / "outputs" / "predictions"
    if not (predictions / "img_a1_class_mask.npy").is_file():
        pytest.skip("inference artefacts not present; run test_end_to_end_pipeline first")

    source = np.load(root / "rift_mask" / "img_a1_mask.npy").astype(bool)
    class_mask = np.load(predictions / "img_a1_class_mask.npy")

    assert (class_mask[~source] == 0).all()
    assert (class_mask[source] != 0).all()
