"""Map edge-level predictions back onto a pixel-level class mask.

Every foreground pixel is assigned the class of its nearest skeleton pixel, so
the output mask has exactly the same support as the Stage-1 mask -- the
classifier redistributes pixels between classes but never invents or deletes
them.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Tuple

import numpy as np

from ..config import ClassesConfig
from ..graph.graph_types import SkeletonGraph, paths_to_pixel_index

logger = logging.getLogger(__name__)

try:
    import cv2

    CV2_ERRORS: Tuple[type, ...] = (AttributeError, cv2.error)
except ImportError:  # pragma: no cover - OpenCV is a hard dependency of RIFT
    cv2 = None  # type: ignore[assignment]
    CV2_ERRORS = (AttributeError, TypeError)


@dataclass
class RasterizationResult:
    """Class mask plus the edge-id map used to build it."""

    class_mask: np.ndarray
    edge_id_map: np.ndarray
    pixel_counts: Dict[int, int]

    def as_dict(self) -> Dict[str, object]:
        return {"pixel_counts": {str(k): int(v) for k, v in self.pixel_counts.items()}}


def nearest_edge_map(
    skeleton_edge_index: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    """Propagate skeleton edge ids to every mask pixel via nearest neighbour.

    Args:
        skeleton_edge_index: ``int32`` array holding the owning edge id at each
            skeleton pixel and ``-1`` elsewhere.
        mask: Boolean foreground mask to fill.

    Returns:
        ``int32`` array with an edge id for every foreground pixel, ``-1``
        outside the mask.
    """
    seeds = skeleton_edge_index >= 0
    if not seeds.any():
        return np.full(mask.shape, -1, dtype=np.int32)

    try:
        source = np.where(seeds, 0, 255).astype(np.uint8)
        _, labels = cv2.distanceTransformWithLabels(
            source, cv2.DIST_L2, 5, labelType=cv2.DIST_LABEL_PIXEL
        )
        lookup = np.full(int(labels.max()) + 1, -1, dtype=np.int32)
        lookup[labels[seeds]] = skeleton_edge_index[seeds]
        propagated = lookup[labels]
    except CV2_ERRORS as error:
        logger.warning("cv2 label propagation unavailable (%s); using scipy", error)
        from scipy import ndimage as ndi

        _, indices = ndi.distance_transform_edt(~seeds, return_indices=True)
        propagated = skeleton_edge_index[indices[0], indices[1]].astype(np.int32)

    propagated = propagated.astype(np.int32)
    propagated[~mask.astype(bool)] = -1
    return propagated


def rasterize_predictions(
    graph: SkeletonGraph,
    mask: np.ndarray,
    edge_labels: Mapping[int, str],
    classes: ClassesConfig,
    uncertain_label: str = "uncertain",
) -> RasterizationResult:
    """Paint predicted edge classes onto the full-resolution mask.

    Edges predicted as ``uncertain`` receive the configured ``ignore`` value so
    that a low-confidence prediction is visibly distinct from a confident one.
    """
    edge_index = paths_to_pixel_index(graph.edges.values(), graph.image_shape)
    propagated = nearest_edge_map(edge_index, mask)

    value_lookup: Dict[int, int] = {}
    for edge_id in graph.edges:
        label = edge_labels.get(edge_id)
        if label is None:
            value_lookup[edge_id] = classes.ignore
        elif label == uncertain_label:
            value_lookup[edge_id] = classes.ignore
        else:
            value_lookup[edge_id] = classes.pixel_value(label)

    maximum_edge_id = max(value_lookup) if value_lookup else -1
    table = np.full(maximum_edge_id + 2, classes.background, dtype=np.uint8)
    for edge_id, value in value_lookup.items():
        table[edge_id] = value

    class_mask = np.full(graph.image_shape, classes.background, dtype=np.uint8)
    foreground = propagated >= 0
    class_mask[foreground] = table[propagated[foreground]]

    values, counts = np.unique(class_mask, return_counts=True)
    pixel_counts = {int(v): int(c) for v, c in zip(values, counts)}
    logger.info("%s: rasterized class pixel counts %s", graph.image_id, pixel_counts)
    return RasterizationResult(class_mask=class_mask, edge_id_map=propagated, pixel_counts=pixel_counts)


def class_mask_to_color(
    class_mask: np.ndarray,
    classes: ClassesConfig,
    palette: Optional[Mapping[int, Tuple[int, int, int]]] = None,
) -> np.ndarray:
    """Colourise a class mask for visual inspection (BGR)."""
    default = {
        classes.background: (0, 0, 0),
        classes.crack: (60, 60, 235),
        classes.craquelure: (235, 180, 60),
        classes.other_line: (60, 200, 235),
        classes.ignore: (170, 170, 170),
    }
    lookup = dict(default)
    if palette:
        lookup.update(palette)
    output = np.zeros((*class_mask.shape, 3), dtype=np.uint8)
    for value, color in lookup.items():
        output[class_mask == value] = color
    return output
