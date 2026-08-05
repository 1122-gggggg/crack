"""Assemble every per-edge feature block into one tabular dataset."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import TopologyConfig
from ..graph.graph_types import SkeletonGraph
from .appearance import AppearancePlanes, appearance_features
from .edge_features import (
    edge_geometry_features,
    edge_probability_features,
    edge_topology_features,
    edge_width_features,
)
from .enclosed_faces import EMPTY_ADJACENCY, edge_face_adjacency, enclosed_faces
from .graph_features import component_features, local_features
from .node_features import aggregate_endpoint_features

logger = logging.getLogger(__name__)

ID_COLUMNS = ("image_id", "panel_id", "component_id", "edge_id", "u", "v", "row", "col")


@dataclass
class FeatureTable:
    """Feature matrix plus the identifier columns needed for grouping and review."""

    frame: pd.DataFrame

    @property
    def feature_columns(self) -> List[str]:
        return [c for c in self.frame.columns if c not in ID_COLUMNS]

    def matrix(self) -> np.ndarray:
        return self.frame[self.feature_columns].to_numpy(dtype=np.float32)

    def save(self, path) -> None:
        from pathlib import Path

        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        if target.suffix == ".parquet":
            try:
                self.frame.to_parquet(target, index=False)
            except (ImportError, ValueError) as error:
                logger.warning("parquet unavailable (%s); writing CSV instead", error)
                self.frame.to_csv(target.with_suffix(".csv"), index=False)
        else:
            self.frame.to_csv(target, index=False)


class FeatureExtractor:
    """Computes handcrafted edge features for one image at a time."""

    def __init__(self, config: TopologyConfig) -> None:
        self.config = config

    def extract(
        self,
        graph: SkeletonGraph,
        mask: Optional[np.ndarray] = None,
        probability: Optional[np.ndarray] = None,
        image: Optional[np.ndarray] = None,
        distance: Optional[np.ndarray] = None,
    ) -> FeatureTable:
        """Build the per-edge feature table for one image.

        Args:
            graph: Skeleton graph in full-image coordinates.
            mask: Boolean line mask, required for enclosed-face features.
            probability: Stage-1 probability map, or ``None``.
            image: BGR image for appearance features, or ``None``.
            distance: Distance transform of the mask; recomputed if omitted.
        """
        graph_config = self.config.graph
        if mask is not None and distance is None:
            import cv2

            distance = cv2.distanceTransform(mask.astype(np.uint8), cv2.DIST_L2, 5)

        strict = tolerant = None
        adjacency_strict: Dict[int, object] = {}
        adjacency_tolerant: Dict[int, object] = {}
        if mask is not None:
            strict = enclosed_faces(mask, graph_config, tolerant=False)
            tolerant = enclosed_faces(mask, graph_config, tolerant=True)
            paths = {edge_id: edge.path for edge_id, edge in graph.edges.items()}
            search_radius = max(1, graph_config.tolerant_face_closing_radius + 1)
            adjacency_strict = edge_face_adjacency(paths, strict, search_radius, distance)
            adjacency_tolerant = edge_face_adjacency(paths, tolerant, search_radius, distance)

        per_component = component_features(graph, graph_config, strict, tolerant)
        planes = (
            AppearancePlanes.from_image(image)
            if image is not None and self.config.appearance.enable
            else None
        )

        rows: List[Dict[str, float]] = []
        for edge in graph.edges.values():
            record: Dict[str, float] = {
                "image_id": graph.image_id,
                "panel_id": graph.panel_id,
                "component_id": edge.component_id,
                "edge_id": edge.edge_id,
                "u": edge.u,
                "v": edge.v,
                "row": float(np.mean(edge.path[:, 0])),
                "col": float(np.mean(edge.path[:, 1])),
            }
            record.update(edge_geometry_features(edge, graph_config))
            record.update(edge_width_features(edge, distance, graph_config))
            record.update(edge_probability_features(edge, probability))
            record.update(edge_topology_features(graph, edge))
            record.update(aggregate_endpoint_features(graph, edge))
            record.update(local_features(graph, edge, graph_config))
            record.update(per_component.get(edge.component_id, {}))
            record.update(
                adjacency_strict.get(edge.edge_id, EMPTY_ADJACENCY).as_dict("strict")  # type: ignore[union-attr]
                if adjacency_strict
                else EMPTY_ADJACENCY.as_dict("strict")
            )
            record.update(
                adjacency_tolerant.get(edge.edge_id, EMPTY_ADJACENCY).as_dict("tolerant")  # type: ignore[union-attr]
                if adjacency_tolerant
                else EMPTY_ADJACENCY.as_dict("tolerant")
            )
            record.update(
                appearance_features(edge, image, distance, self.config.appearance, planes=planes)
            )
            rows.append(record)

        frame = pd.DataFrame(rows)
        if frame.empty:
            frame = pd.DataFrame(columns=list(ID_COLUMNS))
        logger.info(
            "%s: extracted %d edge feature rows x %d columns",
            graph.image_id,
            len(frame),
            max(len(frame.columns) - len(ID_COLUMNS), 0),
        )
        return FeatureTable(frame=frame)
