"""Rendering helpers shared by the review export and the report figures."""
from __future__ import annotations

import logging
from typing import Dict, Iterable, Mapping, Optional, Sequence, Tuple

import numpy as np

from ..graph.graph_types import (
    NODE_ENDPOINT,
    NODE_ISOLATED,
    NODE_JUNCTION,
    NODE_VIRTUAL_LOOP,
    SkeletonGraph,
)

logger = logging.getLogger(__name__)

BGR = Tuple[int, int, int]

CLASS_COLORS: Dict[str, BGR] = {
    "crack": (60, 60, 235),          # red
    "craquelure": (235, 180, 60),    # blue-cyan
    "other_line": (60, 200, 235),    # amber
    "uncertain": (200, 200, 200),    # grey
    "unlabeled": (120, 120, 120),
}
NODE_COLORS: Dict[str, BGR] = {
    NODE_ENDPOINT: (80, 240, 80),
    NODE_JUNCTION: (0, 140, 255),
    NODE_VIRTUAL_LOOP: (255, 0, 255),
    NODE_ISOLATED: (160, 160, 160),
}


def _palette(count: int) -> np.ndarray:
    """Deterministic distinct colours for edge ids (golden-ratio hue walk)."""
    import cv2

    hues = (np.arange(count) * 137.508) % 180.0
    hsv = np.stack([hues, np.full(count, 200.0), np.full(count, 255.0)], axis=1).astype(np.uint8)
    return cv2.cvtColor(hsv[None, ...], cv2.COLOR_HSV2BGR)[0]


def base_canvas(
    shape: Tuple[int, int],
    image: Optional[np.ndarray] = None,
    dim: float = 0.45,
) -> np.ndarray:
    """A dimmed BGR background to draw the graph on top of."""
    if image is None:
        return np.zeros((*shape, 3), dtype=np.uint8)
    canvas = image if image.ndim == 3 else np.repeat(image[..., None], 3, axis=2)
    return (canvas.astype(np.float32) * dim).astype(np.uint8)


def draw_edges(
    canvas: np.ndarray,
    graph: SkeletonGraph,
    colors: Optional[Mapping[int, BGR]] = None,
    thickness: int = 1,
) -> np.ndarray:
    """Paint every edge path; ``colors`` maps edge_id -> BGR."""
    palette = _palette(max(len(graph.edges), 1))
    for index, edge in enumerate(graph.edges.values()):
        color = (
            tuple(int(c) for c in colors[edge.edge_id])
            if colors and edge.edge_id in colors
            else tuple(int(c) for c in palette[index % len(palette)])
        )
        rows, cols = edge.path[:, 0], edge.path[:, 1]
        canvas[rows, cols] = color
        if thickness > 1:
            for offset in range(1, thickness):
                for dr, dc in ((offset, 0), (-offset, 0), (0, offset), (0, -offset)):
                    r = np.clip(rows + dr, 0, canvas.shape[0] - 1)
                    c = np.clip(cols + dc, 0, canvas.shape[1] - 1)
                    canvas[r, c] = color
    return canvas


def draw_nodes(canvas: np.ndarray, graph: SkeletonGraph, radius: int = 2) -> np.ndarray:
    import cv2

    for node in graph.nodes.values():
        color = NODE_COLORS.get(node.node_type, (255, 255, 255))
        cv2.circle(canvas, (int(round(node.col)), int(round(node.row))), radius, color, -1, lineType=cv2.LINE_AA)
    return canvas


def draw_edge_ids(
    canvas: np.ndarray,
    graph: SkeletonGraph,
    edge_ids: Optional[Iterable[int]] = None,
    font_scale: float = 0.4,
) -> np.ndarray:
    """Label edges with their id at the path midpoint."""
    import cv2

    selected = set(edge_ids) if edge_ids is not None else set(graph.edges)
    for edge in graph.edges.values():
        if edge.edge_id not in selected:
            continue
        row, col = edge.path[edge.path.shape[0] // 2]
        position = (int(col) + 3, int(row) - 3)
        cv2.putText(canvas, str(edge.edge_id), position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(
            canvas, str(edge.edge_id), position, cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), 1, cv2.LINE_AA
        )
    return canvas


def class_overlay(
    shape: Tuple[int, int],
    graph: SkeletonGraph,
    edge_labels: Mapping[int, str],
    image: Optional[np.ndarray] = None,
    thickness: int = 2,
) -> np.ndarray:
    """Colour every edge by its class name."""
    canvas = base_canvas(shape, image)
    colors = {
        edge_id: CLASS_COLORS.get(label, CLASS_COLORS["unlabeled"]) for edge_id, label in edge_labels.items()
    }
    return draw_edges(canvas, graph, colors=colors, thickness=thickness)


def legend_rows(names: Sequence[str]) -> Dict[str, BGR]:
    return {name: CLASS_COLORS.get(name, CLASS_COLORS["unlabeled"]) for name in names}
