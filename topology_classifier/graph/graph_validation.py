"""Structural sanity checks for a built :class:`SkeletonGraph`."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from .graph_types import SkeletonGraph

logger = logging.getLogger(__name__)


@dataclass
class GraphValidationReport:
    """Collected problems; ``is_valid`` is False when any error was found."""

    errors: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)
    stats: Dict[str, float] = field(default_factory=dict)

    @property
    def is_valid(self) -> bool:
        return not self.errors

    def to_dict(self) -> Dict[str, object]:
        return {"is_valid": self.is_valid, "errors": self.errors, "warnings": self.warnings, "stats": self.stats}


def validate_graph(graph: SkeletonGraph, *, max_reported: int = 20) -> GraphValidationReport:
    """Verify node references, path continuity and coordinate ranges."""
    report = GraphValidationReport()
    height, width = graph.image_shape

    for edge in graph.edges.values():
        if edge.u not in graph.nodes or edge.v not in graph.nodes:
            _add(report.errors, f"edge {edge.edge_id} references missing node ({edge.u}, {edge.v})", max_reported)
        if edge.path.shape[0] == 0:
            _add(report.errors, f"edge {edge.edge_id} has an empty path", max_reported)
            continue
        rows, cols = edge.path[:, 0], edge.path[:, 1]
        if rows.min() < 0 or cols.min() < 0 or rows.max() >= height or cols.max() >= width:
            _add(report.errors, f"edge {edge.edge_id} leaves the image bounds", max_reported)
        if edge.path.shape[0] > 1:
            steps = np.abs(np.diff(edge.path.astype(np.int64), axis=0)).max(axis=1)
            if int(steps.max()) > 1:
                _add(report.errors, f"edge {edge.edge_id} path is not 8-connected", max_reported)

    seen_pixels: Dict[tuple, int] = {}
    for node in graph.nodes.values():
        if not node.pixels:
            _add(report.warnings, f"node {node.node_id} has no backing pixels", max_reported)
        for pixel in node.pixels:
            if pixel in seen_pixels and seen_pixels[pixel] != node.node_id:
                _add(
                    report.errors,
                    f"pixel {pixel} shared by nodes {seen_pixels[pixel]} and {node.node_id}",
                    max_reported,
                )
            seen_pixels[pixel] = node.node_id

    orphan_nodes = [n.node_id for n in graph.nodes.values() if n.degree == 0 and n.node_type != "isolated"]
    if orphan_nodes:
        _add(report.warnings, f"{len(orphan_nodes)} nodes have no incident edge", max_reported)

    report.stats = graph.summary()
    if not report.is_valid:
        logger.warning("graph validation found %d errors", len(report.errors))
    return report


def _add(bucket: List[str], message: str, limit: int) -> None:
    if len(bucket) < limit:
        bucket.append(message)
    elif len(bucket) == limit:
        bucket.append("... further messages suppressed")
