"""Enclosed faces (cells) of a line network.

Craquelure closes into polygonal cells; structural cracks usually do not. Two
variants are computed:

* **strict**   -- holes of the mask exactly as segmented.
* **tolerant** -- holes after a small morphological closing, so a cell that is
  interrupted by a one or two pixel gap still counts.

The difference between the two is itself informative and is exported as a
feature, so an almost-closed network is not silently treated as open.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import binary_closing, disk

from ..config import GraphConfig

logger = logging.getLogger(__name__)

STRUCTURE_8 = np.ones((3, 3), dtype=bool)


@dataclass
class FaceMap:
    """Labelled enclosed regions plus their areas."""

    labels: np.ndarray
    areas: Dict[int, float] = field(default_factory=dict)

    @property
    def count(self) -> int:
        return len(self.areas)

    @property
    def total_area(self) -> float:
        return float(sum(self.areas.values()))

    def area_list(self) -> List[float]:
        return [self.areas[key] for key in sorted(self.areas)]


def _label_holes(mask: np.ndarray, minimum_area: int) -> FaceMap:
    filled = ndi.binary_fill_holes(mask, structure=STRUCTURE_8)
    if filled is None:
        return FaceMap(labels=np.zeros(mask.shape, dtype=np.int32))
    holes = filled & ~mask
    labels, count = ndi.label(holes, structure=STRUCTURE_8)
    if count == 0:
        return FaceMap(labels=np.zeros(mask.shape, dtype=np.int32))
    sizes = np.bincount(labels.ravel())
    keep = {int(index) for index in range(1, count + 1) if sizes[index] >= minimum_area}
    if len(keep) < count:
        drop = np.array([index not in keep for index in range(len(sizes))], dtype=bool)
        labels[drop[labels]] = 0
    areas = {index: float(sizes[index]) for index in sorted(keep)}
    return FaceMap(labels=labels, areas=areas)


def enclosed_faces(mask: np.ndarray, config: GraphConfig, tolerant: bool = False) -> FaceMap:
    """Label the enclosed background regions of a line mask.

    Args:
        mask: Boolean line mask (not the skeleton -- holes need the full width).
        config: Supplies ``minimum_enclosed_area`` and the closing radius.
        tolerant: When ``True``, close small gaps before hole filling.
    """
    binary = np.ascontiguousarray(mask.astype(bool))
    if tolerant and config.tolerant_face_closing_radius > 0:
        binary = binary_closing(binary, disk(config.tolerant_face_closing_radius))
    return _label_holes(binary, config.minimum_enclosed_area)


@dataclass
class EdgeFaceAdjacency:
    """Per-edge summary of the faces an edge borders."""

    face_count: int
    face_area_mean: float
    face_area_max: float
    face_area_min: float
    borders_face: bool

    def as_dict(self, prefix: str) -> Dict[str, float]:
        return {
            f"{prefix}_face_count": float(self.face_count),
            f"{prefix}_face_area_mean": self.face_area_mean,
            f"{prefix}_face_area_max": self.face_area_max,
            f"{prefix}_face_area_min": self.face_area_min,
            f"{prefix}_borders_face": float(self.borders_face),
        }


EMPTY_ADJACENCY = EdgeFaceAdjacency(
    face_count=0,
    face_area_mean=0.0,
    face_area_max=0.0,
    face_area_min=0.0,
    borders_face=False,
)


def _dilated_face_lookup(face_map: FaceMap, radius: int) -> np.ndarray:
    """Grow each face by ``radius`` so that skeleton pixels fall inside it."""
    if face_map.count == 0:
        return face_map.labels
    grown = ndi.grey_dilation(face_map.labels, size=(2 * radius + 1, 2 * radius + 1))
    return grown.astype(np.int32)


def edge_face_adjacency(
    paths: Dict[int, np.ndarray],
    face_map: FaceMap,
    search_radius: int,
    width_map: Optional[np.ndarray] = None,
) -> Dict[int, EdgeFaceAdjacency]:
    """For every edge, which enclosed faces does its skeleton path touch?

    A skeleton pixel sits in the middle of the line, so the face labels are
    dilated by the local half-width (or ``search_radius``) before lookup.
    """
    if not paths:
        return {}
    if face_map.count == 0:
        return {edge_id: EMPTY_ADJACENCY for edge_id in paths}

    radius = max(1, int(search_radius))
    lookup_stack = [(radius, _dilated_face_lookup(face_map, radius))]
    if width_map is not None:
        extra = int(np.ceil(float(np.nanmax(width_map)) / 2.0)) + 1
        if extra > radius:
            lookup_stack.append((extra, _dilated_face_lookup(face_map, extra)))

    adjacency: Dict[int, EdgeFaceAdjacency] = {}
    for edge_id, path in paths.items():
        found: set[int] = set()
        for _, lookup in lookup_stack:
            values = lookup[path[:, 0], path[:, 1]]
            found.update(int(v) for v in np.unique(values) if v > 0)
            if found:
                break
        if not found:
            adjacency[edge_id] = EMPTY_ADJACENCY
            continue
        areas = [face_map.areas.get(index, 0.0) for index in sorted(found)]
        adjacency[edge_id] = EdgeFaceAdjacency(
            face_count=len(areas),
            face_area_mean=float(np.mean(areas)),
            face_area_max=float(np.max(areas)),
            face_area_min=float(np.min(areas)),
            borders_face=True,
        )
    return adjacency
