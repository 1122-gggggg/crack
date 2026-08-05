"""Run a trained edge classifier over new images."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from ..config import TopologyConfig
from ..io.dataset_adapter import DatasetAdapter
from ..labels.label_types import UNCERTAIN
from ..logging_utils import ErrorJournal
from ..models.baseline import BaselineClassifier
from ..pipeline import ImageArtifacts, build_image_graph
from ..training.dataset import extract_image_features
from ..training.train_gnn import GNNClassifier
from .rasterize import RasterizationResult, class_mask_to_color, rasterize_predictions

logger = logging.getLogger(__name__)

Classifier = Union[BaselineClassifier, GNNClassifier]


@dataclass
class ImagePrediction:
    """Per-edge predictions for one image, plus the rasterized class mask."""

    image_id: str
    frame: pd.DataFrame
    artifacts: ImageArtifacts
    raster: Optional[RasterizationResult] = None
    written: List[Path] = field(default_factory=list)

    def edge_labels(self) -> Dict[int, str]:
        return dict(zip(self.frame["edge_id"].astype(int), self.frame["predicted_label"]))


def _apply_junction_policy(frame: pd.DataFrame, policy: str) -> pd.DataFrame:
    """Optionally soften predictions on edges whose two ends disagree.

    ``uncertain`` (the default) marks an edge as uncertain when its raw
    prediction differs from every neighbouring edge incident to either of its
    terminal nodes.  This is deliberately conservative: an edge is changed
    only when a junction provides contradictory local evidence, while an
    isolated line and an edge that agrees with at least one neighbour retain
    their model prediction. ``keep`` leaves the raw prediction untouched.

    The feature table contains ``u`` and ``v`` graph-node ids.  Older feature
    tables may not have those columns; in that case the policy is skipped with
    a warning rather than guessing from pixel coordinates.
    """
    if policy not in {"uncertain", "keep"}:
        raise ValueError(f"unknown junction_conflict_policy {policy!r}; expected 'uncertain' or 'keep'")
    if policy == "keep" or frame.empty:
        return frame
    required = {"image_id", "edge_id", "u", "v", "raw_label", "predicted_label"}
    missing = required - set(frame.columns)
    if missing:
        logger.warning(
            "junction conflict policy skipped; prediction table lacks %s",
            sorted(missing),
        )
        return frame

    output = frame.copy()
    for _, image_frame in output.groupby("image_id", sort=False, dropna=False):
        node_to_edges: Dict[int, List[int]] = {}
        for row_index, row in image_frame.iterrows():
            for node in {int(row["u"]), int(row["v"])}:
                node_to_edges.setdefault(node, []).append(row_index)

        for row_index, row in image_frame.iterrows():
            raw_label = row["raw_label"]
            if raw_label == UNCERTAIN:
                continue
            neighbours: set[int] = set()
            for node in {int(row["u"]), int(row["v"])}:
                neighbours.update(node_to_edges.get(node, ()))
            neighbours.discard(row_index)
            if not neighbours:
                continue
            neighbour_labels = {
                output.at[index, "raw_label"]
                for index in neighbours
                if output.at[index, "raw_label"] != UNCERTAIN
            }
            if neighbour_labels and raw_label not in neighbour_labels:
                output.at[row_index, "predicted_label"] = UNCERTAIN
    return output


def classify_image(
    config: TopologyConfig,
    model: Classifier,
    artifacts: ImageArtifacts,
    adapter: DatasetAdapter,
) -> pd.DataFrame:
    """Predict a class for every edge of one image.

    Predictions below ``minimum_prediction_confidence`` are relabelled
    ``uncertain`` instead of being forced into a class.
    """
    features = extract_image_features(config, artifacts, adapter)
    if features.empty:
        logger.warning("%s: graph has no edges; nothing to classify", artifacts.image_id)
        return features

    missing = [c for c in model.metadata.feature_names if c not in features.columns]
    if missing:
        raise KeyError(
            f"{artifacts.image_id}: feature table is missing {len(missing)} column(s) the model expects, "
            f"e.g. {missing[:5]}; rebuild features with the same config"
        )
    if isinstance(model, GNNClassifier):
        probabilities = model.predict_proba(features)
    else:
        matrix = features[model.metadata.feature_names].to_numpy(dtype=np.float32)
        probabilities = model.predict_proba(matrix)
    indices = np.argmax(probabilities, axis=1)
    confidence = probabilities.max(axis=1)

    class_names = model.metadata.class_names
    predicted = [class_names[int(i)] for i in indices]
    threshold = config.model.minimum_prediction_confidence
    labels = [
        name if score >= threshold else UNCERTAIN for name, score in zip(predicted, confidence)
    ]

    output = features[
        ["image_id", "panel_id", "component_id", "edge_id", "u", "v", "row", "col", "length_px"]
    ].copy()
    output["predicted_label"] = labels
    output["raw_label"] = predicted
    output["confidence"] = confidence
    for index, name in enumerate(class_names):
        output[f"prob_{name}"] = probabilities[:, index]
    output = _apply_junction_policy(output, config.model.junction_conflict_policy)

    counts = output["predicted_label"].value_counts().to_dict()
    logger.info("%s: predicted %s (abstained=%d)", artifacts.image_id, counts, counts.get(UNCERTAIN, 0))
    return output


def infer_dataset(
    config: TopologyConfig,
    model_path: Path,
    adapter: Optional[DatasetAdapter] = None,
    journal: Optional[ErrorJournal] = None,
    output_dir: Optional[Path] = None,
    limit: Optional[int] = None,
    write_masks: bool = True,
    model_type: str = "auto",
) -> List[ImagePrediction]:
    """Classify every discoverable image and write per-image artefacts."""
    import cv2

    adapter = adapter or DatasetAdapter(config)
    journal = journal or ErrorJournal(config.output_dir / config.runtime.errors_filename)
    output_dir = Path(output_dir) if output_dir else config.output_dir / "predictions"
    output_dir.mkdir(parents=True, exist_ok=True)

    model_path = Path(model_path)
    if model_type not in {"auto", "baseline", "gnn"}:
        raise ValueError("model_type must be 'auto', 'baseline', or 'gnn'")
    resolved_model_type = model_type
    if resolved_model_type == "auto":
        resolved_model_type = "gnn" if model_path.suffix.lower() in {".pt", ".pth"} else "baseline"
    if resolved_model_type == "gnn":
        model: Classifier = GNNClassifier.load(model_path)
    else:
        model = BaselineClassifier.load(model_path, config.model, config.training)
    expected_hash = config.config_hash(scope="features")
    if model.metadata.config_hash and model.metadata.config_hash != expected_hash:
        logger.warning(
            "model was trained with feature config hash %s but the current config hashes to %s; "
            "features may be incompatible",
            model.metadata.config_hash,
            expected_hash,
        )

    records = adapter.records()
    if limit is not None:
        records = records[:limit]

    predictions: List[ImagePrediction] = []
    for index, record in enumerate(records, start=1):
        logger.info("[%d/%d] inferring %s", index, len(records), record.image_id)
        try:
            artifacts = build_image_graph(config, record, adapter)
            frame = classify_image(config, model, artifacts, adapter)
            prediction = ImagePrediction(image_id=record.image_id, frame=frame, artifacts=artifacts)

            csv_path = output_dir / f"{record.image_id}_edge_predictions.csv"
            frame.to_csv(csv_path, index=False)
            prediction.written.append(csv_path)

            if write_masks and not frame.empty:
                raster = rasterize_predictions(
                    artifacts.graph, artifacts.mask, prediction.edge_labels(), config.classes
                )
                prediction.raster = raster
                mask_path = output_dir / f"{record.image_id}_class_mask.png"
                color_path = output_dir / f"{record.image_id}_class_color.png"
                cv2.imwrite(str(mask_path), raster.class_mask)
                cv2.imwrite(str(color_path), class_mask_to_color(raster.class_mask, config.classes))
                np.save(output_dir / f"{record.image_id}_class_mask.npy", raster.class_mask)
                prediction.written.extend([mask_path, color_path])

            predictions.append(prediction)
        except (OSError, ValueError, KeyError, MemoryError) as error:
            journal.record(record.image_id, stage="infer", error=error)

    logger.info("inference complete for %d/%d image(s)", len(predictions), len(records))
    return predictions
