"""Turn a pixel-level annotation mask into edge-level labels.

Rules enforced here:

* Pixels near a junction are ambiguous (two classes physically meet there), so
  they are excluded within ``junction_ignore_radius``.
* An edge is labelled only when one class holds at least ``edge_label_purity``
  of its labelled samples. Otherwise it becomes ``uncertain`` -- never silently
  rounded to the majority class.
* An edge with no annotated pixel becomes ``unlabeled`` and is excluded from
  training rather than treated as background.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from ..config import ClassesConfig, GraphConfig, LabelsConfig
from ..graph.graph_types import NODE_JUNCTION, SkeletonGraph
from .label_types import UNCERTAIN, UNLABELED, EdgeLabel, LabelVocabulary

logger = logging.getLogger(__name__)


def junction_exclusion_mask(graph: SkeletonGraph, radius: int) -> np.ndarray:
    """Boolean mask of pixels that are too close to a junction to trust."""
    mask = np.zeros(graph.image_shape, dtype=bool)
    if radius <= 0:
        return mask
    for node in graph.nodes.values():
        if node.node_type != NODE_JUNCTION:
            continue
        for row, col in node.pixels or ((int(round(node.row)), int(round(node.col))),):
            mask[
                max(0, row - radius) : min(mask.shape[0], row + radius + 1),
                max(0, col - radius) : min(mask.shape[1], col + radius + 1),
            ] = True
    return mask


def _sample_positions(
    path: np.ndarray,
    width_radius: Optional[np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Skeleton path pixels, optionally thickened to the local line width."""
    if width_radius is None:
        return path[:, 0], path[:, 1]
    rows: List[np.ndarray] = []
    cols: List[np.ndarray] = []
    for row, col in path:
        radius = int(round(float(width_radius[row, col])))
        if radius <= 0:
            rows.append(np.asarray([row]))
            cols.append(np.asarray([col]))
            continue
        offsets = np.arange(-radius, radius + 1)
        grid_r, grid_c = np.meshgrid(offsets, offsets, indexing="ij")
        keep = (grid_r**2 + grid_c**2) <= radius**2
        rows.append(row + grid_r[keep])
        cols.append(col + grid_c[keep])
    return np.concatenate(rows), np.concatenate(cols)


def labels_from_pixel_mask(
    graph: SkeletonGraph,
    label_mask: np.ndarray,
    classes: ClassesConfig,
    labels_config: LabelsConfig,
    graph_config: GraphConfig,
    distance: Optional[np.ndarray] = None,
) -> List[EdgeLabel]:
    """Assign one label per edge by majority vote over its annotated pixels.

    Args:
        graph: Skeleton graph for the image.
        label_mask: Integer mask holding the class pixel values from the config.
        classes: Class value definitions.
        labels_config: Purity threshold and junction exclusion radius.
        graph_config: Unused today but kept so callers pass one config bundle.
        distance: Distance transform; when given (and enabled in the config) the
            annotation is sampled across the full line width, not just the
            one-pixel skeleton.

    Raises:
        ValueError: If ``label_mask`` does not match the graph's image shape.
    """
    if label_mask.shape != graph.image_shape:
        raise ValueError(
            f"label mask shape {label_mask.shape} != image shape {graph.image_shape}"
        )
    del graph_config  # kept for signature symmetry with the other label sources

    vocabulary = LabelVocabulary(classes)
    excluded = junction_exclusion_mask(graph, labels_config.junction_ignore_radius)
    width_radius = distance if labels_config.label_sample_uses_width else None
    height, width = graph.image_shape

    results: List[EdgeLabel] = []
    for edge in graph.edges.values():
        rows, cols = _sample_positions(edge.path, width_radius)
        inside = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
        rows, cols = rows[inside], cols[inside]
        if rows.size == 0:
            results.append(EdgeLabel(edge.edge_id, UNLABELED, 0.0, 0, "no_samples"))
            continue

        keep = ~excluded[rows, cols]
        rows, cols = rows[keep], cols[keep]
        if rows.size == 0:
            results.append(EdgeLabel(edge.edge_id, UNCERTAIN, 0.0, 0, "junction_only"))
            continue

        values = label_mask[rows, cols]
        counts: Dict[str, int] = {}
        for value in np.unique(values):
            name = vocabulary.name_of_value(int(value))
            if name is None:
                continue
            counts[name] = int(np.count_nonzero(values == value))

        total = sum(counts.values())
        if total == 0:
            results.append(EdgeLabel(edge.edge_id, UNLABELED, 0.0, 0, "no_annotation"))
            continue

        best_name = max(counts, key=lambda name: counts[name])
        purity = counts[best_name] / total
        if purity >= labels_config.edge_label_purity:
            results.append(EdgeLabel(edge.edge_id, best_name, purity, total, "majority"))
        else:
            results.append(EdgeLabel(edge.edge_id, UNCERTAIN, purity, total, "mixed_below_purity"))

    _log_distribution(graph.image_id, results)
    return results


def _log_distribution(image_id: str, labels: List[EdgeLabel]) -> None:
    distribution: Dict[str, int] = {}
    for item in labels:
        distribution[item.label] = distribution.get(item.label, 0) + 1
    logger.info("%s: edge label distribution %s", image_id, distribution)


def labels_to_frame(image_id: str, panel_id: str, labels: List[EdgeLabel]) -> pd.DataFrame:
    """Tabular form used to join labels onto the feature table."""
    if not labels:
        return pd.DataFrame(
            columns=["image_id", "panel_id", "edge_id", "label", "label_purity", "label_pixel_count", "label_reason"]
        )
    frame = pd.DataFrame([item.as_dict() for item in labels])
    frame.insert(0, "panel_id", panel_id)
    frame.insert(0, "image_id", image_id)
    return frame


def rasterize_edge_labels(
    graph: SkeletonGraph,
    labels: Dict[int, str],
    classes: ClassesConfig,
    distance: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Paint edge labels back onto a pixel mask (debug view of the supervision)."""
    vocabulary = LabelVocabulary(classes)
    output = np.full(graph.image_shape, classes.background, dtype=np.uint8)
    for edge in graph.edges.values():
        name = labels.get(edge.edge_id)
        if name is None or name == UNLABELED:
            continue
        value = vocabulary.value_of_name(name)
        rows, cols = _sample_positions(edge.path, distance)
        inside = (
            (rows >= 0) & (rows < output.shape[0]) & (cols >= 0) & (cols < output.shape[1])
        )
        output[rows[inside], cols[inside]] = value
    return output


def dilate_thin_labels(label_image: np.ndarray, background: int, radius: int) -> np.ndarray:
    """Thicken a label image for human viewing without changing class values."""
    if radius <= 0:
        return label_image
    result = label_image.copy()
    for value in np.unique(label_image):
        if value == background:
            continue
        grown = ndi.binary_dilation(label_image == value, iterations=radius)
        result[grown & (result == background)] = value
    return result
