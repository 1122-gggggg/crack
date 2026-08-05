"""Build the edge-level feature dataset and attach supervision to it."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import TopologyConfig
from ..features.extractor import ID_COLUMNS, FeatureExtractor
from ..io.dataset_adapter import DatasetAdapter
from ..labels.csv_edge_labels import read_edge_annotations, trainable_subset
from ..labels.label_types import UNLABELED, LabelVocabulary
from ..labels.pixel_to_edge_labels import labels_from_pixel_mask, labels_to_frame
from ..logging_utils import ErrorJournal
from ..pipeline import ImageArtifacts, build_image_graph

logger = logging.getLogger(__name__)

LABEL_COLUMNS = ("label", "label_purity", "label_pixel_count", "label_reason")


@dataclass
class DatasetBuildResult:
    """The assembled feature table plus per-image bookkeeping."""

    features: pd.DataFrame
    per_image: List[Dict[str, object]] = field(default_factory=list)
    failed_image_ids: List[str] = field(default_factory=list)

    @property
    def feature_columns(self) -> List[str]:
        skip = set(ID_COLUMNS) | set(LABEL_COLUMNS)
        return [c for c in self.features.columns if c not in skip]

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.suffix == ".parquet":
            try:
                self.features.to_parquet(path, index=False)
                return path
            except (ImportError, ValueError) as error:
                logger.warning("parquet unavailable (%s); writing CSV instead", error)
                path = path.with_suffix(".csv")
        self.features.to_csv(path, index=False)
        return path


def _load_label_mask(path: Path) -> np.ndarray:
    if path.suffix == ".npy":
        return np.load(path)
    import cv2

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise OSError(f"failed to read label mask {path}")
    return array[..., 0] if array.ndim == 3 else array


def extract_image_features(
    config: TopologyConfig,
    artifacts: ImageArtifacts,
    adapter: DatasetAdapter,
) -> pd.DataFrame:
    """Feature rows for one image, with pixel-derived labels when available."""
    image = adapter.load_image(artifacts.record) if config.appearance.enable else None
    extractor = FeatureExtractor(config)
    table = extractor.extract(
        artifacts.graph,
        mask=artifacts.mask,
        probability=artifacts.probability,
        image=image,
        distance=artifacts.distance(),
    )
    frame = table.frame
    if frame.empty:
        return frame

    label_path = artifacts.record.label_path
    if label_path is not None:
        label_mask = _load_label_mask(Path(label_path))
        edge_labels = labels_from_pixel_mask(
            artifacts.graph,
            label_mask,
            config.classes,
            config.labels,
            config.graph,
            distance=artifacts.distance(),
        )
        label_frame = labels_to_frame(artifacts.image_id, artifacts.record.panel_id, edge_labels)
        frame = frame.merge(
            label_frame.drop(columns=["panel_id"]), on=["image_id", "edge_id"], how="left"
        )
        frame["label"] = frame["label"].fillna(UNLABELED)
    return frame


def build_feature_dataset(
    config: TopologyConfig,
    adapter: Optional[DatasetAdapter] = None,
    journal: Optional[ErrorJournal] = None,
    limit: Optional[int] = None,
    use_cache: Optional[bool] = None,
) -> DatasetBuildResult:
    """Run graph construction and feature extraction over the whole dataset."""
    adapter = adapter or DatasetAdapter(config)
    journal = journal or ErrorJournal(config.output_dir / config.runtime.errors_filename)
    records = adapter.records()
    if limit is not None:
        records = records[:limit]

    frames: List[pd.DataFrame] = []
    per_image: List[Dict[str, object]] = []
    failed: List[str] = []
    debug_root = config.output_dir / "debug" if config.preprocessing.save_debug_masks else None

    for index, record in enumerate(records, start=1):
        logger.info("[%d/%d] features for %s", index, len(records), record.image_id)
        try:
            artifacts = build_image_graph(config, record, adapter, use_cache=use_cache, debug_dir=debug_root)
            frame = extract_image_features(config, artifacts, adapter)
            frames.append(frame)
            per_image.append(
                {
                    "image_id": record.image_id,
                    "panel_id": record.panel_id,
                    "edge_count": int(len(frame)),
                    "probability_available": artifacts.probability_available,
                    **{k: v for k, v in artifacts.stats.items() if isinstance(v, (int, float, str, bool))},
                }
            )
        except (OSError, ValueError, KeyError, MemoryError) as error:
            journal.record(record.image_id, stage="build_features", error=error)
            failed.append(record.image_id)

    features = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame(columns=list(ID_COLUMNS))
    logger.info(
        "feature dataset: %d rows from %d/%d image(s)", len(features), len(frames), len(records)
    )
    return DatasetBuildResult(features=features, per_image=per_image, failed_image_ids=failed)


def attach_csv_labels(features: pd.DataFrame, config: TopologyConfig) -> pd.DataFrame:
    """Overlay human CSV annotations on top of any pixel-derived labels."""
    if not config.data.edge_annotations_csv:
        return features
    annotations = read_edge_annotations(config.data.edge_annotations_csv, config.classes)
    columns = ["image_id", "edge_id", "label"]
    if "confidence" in annotations.columns:
        columns.append("confidence")
    merged = features.merge(
        annotations[columns].rename(columns={"label": "csv_label"}),
        on=["image_id", "edge_id"],
        how="left",
    )
    if "label" in merged.columns:
        merged["label"] = merged["csv_label"].where(merged["csv_label"].notna(), merged["label"])
    else:
        merged["label"] = merged["csv_label"]
    merged = merged.drop(columns=["csv_label"])
    merged["label"] = merged["label"].fillna(UNLABELED)
    logger.info("attached CSV labels: %s", merged["label"].value_counts().to_dict())
    return merged


def prepare_training_frame(features: pd.DataFrame, config: TopologyConfig) -> tuple[pd.DataFrame, List[str]]:
    """Filter to trainable rows and map label names to contiguous class indices.

    Raises:
        ValueError: If no trainable rows remain -- training on fabricated labels
            is never an acceptable fallback.
    """
    if "label" not in features.columns:
        raise ValueError(
            "feature table has no 'label' column: provide labels_dir masks or edge_annotations_csv"
        )
    vocabulary = LabelVocabulary(config.classes)
    trainable = trainable_subset(features, config.classes)
    if trainable.empty:
        raise ValueError(
            "no trainable edges after filtering (all rows are unlabeled/uncertain); "
            "annotate edges with `export-edge-review` first"
        )
    trainable = trainable.copy()
    trainable["y"] = trainable["label"].map(vocabulary.index_of).astype(int)
    present = sorted(trainable["label"].unique())
    logger.info("training rows: %d, classes present: %s", len(trainable), present)
    return trainable, vocabulary.names


def feature_matrix(frame: pd.DataFrame, feature_columns: List[str]) -> np.ndarray:
    missing = [c for c in feature_columns if c not in frame.columns]
    if missing:
        raise KeyError(f"feature columns missing from frame: {missing[:10]}")
    return frame[feature_columns].to_numpy(dtype=np.float32)


def numeric_feature_columns(frame: pd.DataFrame) -> List[str]:
    """All numeric columns that are neither identifiers nor label bookkeeping."""
    skip = set(ID_COLUMNS) | set(LABEL_COLUMNS) | {"y", "confidence"}
    return [
        column
        for column in frame.columns
        if column not in skip and pd.api.types.is_numeric_dtype(frame[column])
    ]
