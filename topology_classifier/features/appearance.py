"""Photometric descriptors sampled along an edge and on both of its sides.

A structural crack is usually a dark gap with a genuine intensity drop relative
to the surrounding paint; craquelure on a painted panel is often a thin, lower
contrast fissure whose two sides look alike. Sampling the line and both flanks
separately makes that difference measurable, and the left/right asymmetry also
catches shadows and lifted flakes.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np

from ..config import AppearanceConfig
from ..graph.graph_types import SkeletonEdge

logger = logging.getLogger(__name__)

NAN = float("nan")
APPEARANCE_KEYS = (
    "line_gray_mean",
    "line_gray_std",
    "side_gray_mean",
    "contrast",
    "contrast_normalized",
    "side_asymmetry",
    "line_saturation_mean",
    "side_saturation_mean",
    "patch_gray_std",
    "sample_count",
)


def _placeholder() -> Dict[str, float]:
    features = {key: NAN for key in APPEARANCE_KEYS}
    features["sample_count"] = 0.0
    features["appearance_available"] = 0.0
    return features


@dataclass(frozen=True)
class AppearancePlanes:
    """Grey and saturation planes derived once per image.

    Converting the colour space per edge dominates the runtime on a 40 Mpixel
    panel, so the planes are built once and kept as ``uint8``; only the handful
    of sampled values is promoted to float.
    """

    gray: np.ndarray
    saturation: Optional[np.ndarray]

    @classmethod
    def from_image(cls, image: np.ndarray) -> "AppearancePlanes":
        import cv2

        if image.ndim == 3:
            return cls(
                gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                saturation=cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[..., 1],
            )
        return cls(gray=image, saturation=None)


def _sample(gray: np.ndarray, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    height, width = gray.shape
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    if not valid.any():
        return np.empty(0, dtype=np.float64)
    return gray[rows[valid], cols[valid]].astype(np.float64)


def _unit_normals(path: np.ndarray, indices: np.ndarray) -> np.ndarray:
    """Unit normal (row, col) at each sampled index, from a centred difference."""
    count = path.shape[0]
    previous = np.clip(indices - 1, 0, count - 1)
    following = np.clip(indices + 1, 0, count - 1)
    tangents = path[following].astype(np.float64) - path[previous].astype(np.float64)
    norms = np.hypot(tangents[:, 0], tangents[:, 1])
    norms[norms < 1e-9] = 1.0
    tangents /= norms[:, None]
    return np.stack([tangents[:, 1], -tangents[:, 0]], axis=1)  # rotate by 90 deg


def appearance_features(
    edge: SkeletonEdge,
    image: Optional[np.ndarray],
    distance: Optional[np.ndarray],
    config: AppearanceConfig,
    planes: Optional[AppearancePlanes] = None,
) -> Dict[str, float]:
    """Intensity contrast between the line and its two flanks.

    Args:
        edge: The edge to describe.
        image: BGR ``uint8`` image, or ``None`` when no RGB source exists.
        distance: Distance transform used to place the flank samples outside
            the line; falls back to a fixed offset when ``None``.
        config: Sampling geometry.
        planes: Pre-converted grey/saturation planes; built from ``image`` when
            omitted. Pass them in when looping over many edges.

    Returns:
        Feature dict; all values are NaN and ``appearance_available`` is ``0``
        when the image is missing, so absent appearance is never confused with
        zero contrast.
    """
    if not config.enable or (image is None and planes is None):
        return _placeholder()

    if planes is None:
        planes = AppearancePlanes.from_image(image)  # type: ignore[arg-type]
    gray = planes.gray
    saturation = planes.saturation

    path = edge.path
    count = path.shape[0]
    if count == 0:
        return _placeholder()
    stride = max(1, int(np.ceil(count / max(config.max_samples_per_edge, 1))))
    indices = np.arange(0, count, stride)
    points = path[indices]

    if distance is not None:
        offsets = distance[points[:, 0], points[:, 1]].astype(np.float64) + config.side_offset_extra_px
    else:
        offsets = np.full(points.shape[0], 1.0 + config.side_offset_extra_px)

    normals = _unit_normals(path, indices)
    base = points.astype(np.float64)
    left = np.rint(base + normals * offsets[:, None]).astype(np.int32)
    right = np.rint(base - normals * offsets[:, None]).astype(np.int32)

    line_values = _sample(gray, points[:, 0], points[:, 1])
    left_values = _sample(gray, left[:, 0], left[:, 1])
    right_values = _sample(gray, right[:, 0], right[:, 1])
    if line_values.size == 0 or (left_values.size == 0 and right_values.size == 0):
        return _placeholder()

    side_values = np.concatenate([v for v in (left_values, right_values) if v.size])
    line_mean = float(line_values.mean())
    side_mean = float(side_values.mean())
    contrast = side_mean - line_mean
    left_mean = float(left_values.mean()) if left_values.size else NAN
    right_mean = float(right_values.mean()) if right_values.size else NAN
    denominator = abs(left_mean) + abs(right_mean)

    radius = max(1, config.patch_radius_px)
    row0 = max(0, int(points[:, 0].min()) - radius)
    row1 = min(gray.shape[0], int(points[:, 0].max()) + radius + 1)
    col0 = max(0, int(points[:, 1].min()) - radius)
    col1 = min(gray.shape[1], int(points[:, 1].max()) + radius + 1)
    patch = gray[row0:row1, col0:col1].astype(np.float64)

    features: Dict[str, float] = {
        "line_gray_mean": line_mean,
        "line_gray_std": float(line_values.std()),
        "side_gray_mean": side_mean,
        "contrast": float(contrast),
        "contrast_normalized": float(contrast / (side_mean + 1e-6)),
        "side_asymmetry": float(abs(left_mean - right_mean) / (denominator + 1e-6))
        if denominator > 0 and not np.isnan(left_mean) and not np.isnan(right_mean)
        else NAN,
        "patch_gray_std": float(patch.std()) if patch.size else NAN,
        "sample_count": float(points.shape[0]),
        "appearance_available": 1.0,
    }

    if saturation is not None:
        line_saturation = _sample(saturation, points[:, 0], points[:, 1])
        side_saturation = np.concatenate(
            [v for v in (_sample(saturation, left[:, 0], left[:, 1]), _sample(saturation, right[:, 0], right[:, 1])) if v.size]
            or [np.empty(0)]
        )
        features["line_saturation_mean"] = float(line_saturation.mean()) if line_saturation.size else NAN
        features["side_saturation_mean"] = float(side_saturation.mean()) if side_saturation.size else NAN
    else:
        features["line_saturation_mean"] = NAN
        features["side_saturation_mean"] = NAN

    return features
