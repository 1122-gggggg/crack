"""Conservative endpoint-to-endpoint gap repair.

Bridging gaps changes topology, which is exactly what the downstream classifier
measures, so this step is disabled by default and every bridge must pass three
independent gates: distance, tangent agreement and probability support along the
candidate segment. Each accepted bridge is recorded so the effect can be audited.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from skimage.morphology import skeletonize as sk_skeletonize

from ..config import PreprocessingConfig
from ..graph.skeletonize import neighbor_count

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GapBridge:
    """One accepted bridge between two skeleton endpoints."""

    start: Tuple[int, int]
    end: Tuple[int, int]
    distance_px: float
    angle_start_deg: float
    angle_end_deg: float
    mean_probability: float

    def as_dict(self) -> Dict[str, float | Tuple[int, int]]:
        return {
            "start": self.start,
            "end": self.end,
            "distance_px": round(self.distance_px, 3),
            "angle_start_deg": round(self.angle_start_deg, 2),
            "angle_end_deg": round(self.angle_end_deg, 2),
            "mean_probability": round(self.mean_probability, 4),
        }


@dataclass
class GapRepairResult:
    mask: np.ndarray
    enabled: bool
    bridges: List[GapBridge] = field(default_factory=list)
    rejected: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "enabled": self.enabled,
            "bridge_count": len(self.bridges),
            "bridges": [b.as_dict() for b in self.bridges],
            "rejected": self.rejected,
        }


def _endpoint_coordinates(skeleton: np.ndarray) -> np.ndarray:
    degree = neighbor_count(skeleton)
    rows, cols = np.nonzero(skeleton & (degree == 1))
    return np.stack([rows, cols], axis=1).astype(np.int32) if rows.size else np.empty((0, 2), np.int32)


def _walk_from_endpoint(skeleton: np.ndarray, start: Tuple[int, int], window: int) -> List[Tuple[int, int]]:
    """Follow the skeleton inwards for up to ``window`` pixels."""
    height, width = skeleton.shape
    path = [start]
    visited = {start}
    current = start
    for _ in range(window):
        row, col = current
        nxt: Optional[Tuple[int, int]] = None
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                r, c = row + dr, col + dc
                if 0 <= r < height and 0 <= c < width and skeleton[r, c] and (r, c) not in visited:
                    nxt = (r, c)
                    break
            if nxt is not None:
                break
        if nxt is None:
            break
        visited.add(nxt)
        path.append(nxt)
        current = nxt
    return path


def _outward_tangent(path: Sequence[Tuple[int, int]]) -> Optional[np.ndarray]:
    """Unit vector pointing away from the skeleton at the endpoint."""
    if len(path) < 2:
        return None
    head = np.asarray(path[0], dtype=np.float64)
    tail = np.asarray(path[-1], dtype=np.float64)
    vector = head - tail
    norm = float(np.linalg.norm(vector))
    if norm < 1e-6:
        return None
    return vector / norm


def _angle_between(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(np.dot(a, b), -1.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def _segment_pixels(start: Tuple[int, int], end: Tuple[int, int]) -> np.ndarray:
    steps = int(max(abs(end[0] - start[0]), abs(end[1] - start[1]))) + 1
    rows = np.rint(np.linspace(start[0], end[0], steps)).astype(np.int32)
    cols = np.rint(np.linspace(start[1], end[1], steps)).astype(np.int32)
    return np.stack([rows, cols], axis=1)


def repair_gaps(
    mask: np.ndarray,
    config: PreprocessingConfig,
    probability: Optional[np.ndarray] = None,
) -> GapRepairResult:
    """Bridge short, well-aligned, probability-supported gaps between endpoints."""
    binary = np.ascontiguousarray(mask.astype(bool))
    if not config.enable_gap_repair:
        return GapRepairResult(mask=binary, enabled=False)
    if not binary.any():
        return GapRepairResult(mask=binary, enabled=True)

    skeleton = sk_skeletonize(binary)
    endpoints = _endpoint_coordinates(skeleton)
    if len(endpoints) < 2:
        return GapRepairResult(mask=binary, enabled=True)

    tangents: Dict[int, np.ndarray] = {}
    for index, point in enumerate(endpoints):
        tangent = _outward_tangent(
            _walk_from_endpoint(skeleton, (int(point[0]), int(point[1])), config.gap_tangent_window)
        )
        if tangent is not None:
            tangents[index] = tangent

    deltas = endpoints[:, None, :].astype(np.float64) - endpoints[None, :, :].astype(np.float64)
    distances = np.hypot(deltas[..., 0], deltas[..., 1])
    np.fill_diagonal(distances, np.inf)

    candidates = [
        (float(distances[i, j]), i, j)
        for i, j in zip(*np.nonzero(distances <= config.max_gap_pixels))
        if i < j
    ]
    candidates.sort()

    used: set[int] = set()
    rejected: Dict[str, int] = {}
    bridges: List[GapBridge] = []
    repaired = binary.copy()

    for distance, i, j in candidates:
        if i in used or j in used:
            continue
        if i not in tangents or j not in tangents:
            rejected["no_tangent"] = rejected.get("no_tangent", 0) + 1
            continue
        start = (int(endpoints[i][0]), int(endpoints[i][1]))
        end = (int(endpoints[j][0]), int(endpoints[j][1]))
        direction = np.asarray(end, dtype=np.float64) - np.asarray(start, dtype=np.float64)
        norm = float(np.linalg.norm(direction))
        if norm < 1e-6:
            continue
        direction /= norm

        angle_start = _angle_between(tangents[i], direction)
        angle_end = _angle_between(tangents[j], -direction)
        if angle_start > config.max_angle_difference_deg or angle_end > config.max_angle_difference_deg:
            rejected["angle"] = rejected.get("angle", 0) + 1
            continue

        pixels = _segment_pixels(start, end)
        if probability is not None:
            # Score the gap interior only: the two endpoints sit on the skeleton
            # and would otherwise drag the mean above the threshold on their own.
            interior = pixels[1:-1] if pixels.shape[0] > 2 else pixels
            values = probability[interior[:, 0], interior[:, 1]].astype(np.float64)
            mean_probability = float(values.mean()) if values.size else 0.0
            if mean_probability < config.minimum_gap_confidence:
                rejected["confidence"] = rejected.get("confidence", 0) + 1
                continue
        else:
            mean_probability = float("nan")

        repaired[pixels[:, 0], pixels[:, 1]] = True
        used.update({i, j})
        bridges.append(
            GapBridge(
                start=start,
                end=end,
                distance_px=distance,
                angle_start_deg=angle_start,
                angle_end_deg=angle_end,
                mean_probability=mean_probability,
            )
        )

    logger.info("gap repair: %d bridges accepted, rejected=%s", len(bridges), rejected or "none")
    return GapRepairResult(mask=repaired, enabled=True, bridges=bridges, rejected=rejected)


def gap_repair_debug_image(before: np.ndarray, after: np.ndarray) -> np.ndarray:
    """RGB image: white = original mask, red = pixels added by gap repair."""
    before_bool = before.astype(bool)
    added = after.astype(bool) & ~before_bool
    image = np.zeros((*before_bool.shape, 3), dtype=np.uint8)
    image[before_bool] = (255, 255, 255)
    image[added] = (0, 0, 255)  # BGR red for cv2.imwrite
    return image
