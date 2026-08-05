"""Noise removal that does not punish thin structures.

Real cracks are narrow, so filtering on component *area* would delete them. The
criteria used here are skeleton length, probability confidence, bounding-box
diagonal and elongation, all configurable.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize as sk_skeletonize

from ..config import PreprocessingConfig

logger = logging.getLogger(__name__)

STRUCTURE_8 = np.ones((3, 3), dtype=bool)


@dataclass
class CleanupResult:
    mask: np.ndarray
    removed_component_count: int = 0
    kept_component_count: int = 0
    removal_reasons: Dict[str, int] = field(default_factory=dict)
    removed_pixel_count: int = 0

    def as_dict(self) -> Dict[str, object]:
        return {
            "removed_component_count": self.removed_component_count,
            "kept_component_count": self.kept_component_count,
            "removed_pixel_count": self.removed_pixel_count,
            "removal_reasons": self.removal_reasons,
        }


def _elongation(rows: np.ndarray, cols: np.ndarray) -> float:
    if rows.size < 2:
        return 0.0
    coords = np.stack([rows.astype(np.float64), cols.astype(np.float64)], axis=1)
    coords -= coords.mean(axis=0)
    cov = np.cov(coords, rowvar=False)
    eigenvalues = np.linalg.eigvalsh(cov)
    major = float(max(eigenvalues[-1], 0.0))
    minor = float(max(eigenvalues[0], 0.0))
    return float(np.sqrt(major / (minor + 1e-9)))


def remove_noise_components(
    mask: np.ndarray,
    config: PreprocessingConfig,
    probability: Optional[np.ndarray] = None,
) -> CleanupResult:
    """Drop components that are too short, too faint or too blob-like."""
    binary = np.ascontiguousarray(mask.astype(bool))
    if not binary.any():
        return CleanupResult(mask=binary)

    labels, count = ndi.label(binary, structure=STRUCTURE_8)
    keep = np.zeros(count + 1, dtype=bool)
    reasons: Dict[str, int] = {}
    removed_pixels = 0

    objects = ndi.find_objects(labels)
    for index, slices in enumerate(objects, start=1):
        if slices is None:
            continue
        sub = labels[slices] == index
        skeleton = sk_skeletonize(sub)
        skeleton_length = float(skeleton.sum())
        height = slices[0].stop - slices[0].start
        width = slices[1].stop - slices[1].start
        bbox_diagonal = float(np.hypot(height, width))
        rows, cols = np.nonzero(sub)

        reason: Optional[str] = None
        if skeleton_length < config.minimum_skeleton_length:
            reason = "skeleton_length"
        elif bbox_diagonal < config.minimum_bbox_diagonal:
            reason = "bbox_diagonal"
        elif config.minimum_elongation > 0 and _elongation(rows, cols) < config.minimum_elongation:
            reason = "elongation"
        elif probability is not None and config.minimum_component_confidence > 0:
            values = probability[slices][sub]
            if values.size and float(values.mean()) < config.minimum_component_confidence:
                reason = "confidence"

        if reason is None:
            keep[index] = True
        else:
            reasons[reason] = reasons.get(reason, 0) + 1
            removed_pixels += int(sub.sum())

    cleaned = keep[labels]
    result = CleanupResult(
        mask=cleaned,
        removed_component_count=int(count - keep.sum()),
        kept_component_count=int(keep.sum()),
        removal_reasons=reasons,
        removed_pixel_count=removed_pixels,
    )
    logger.info(
        "cleanup: kept %d/%d components (%d px removed, reasons=%s)",
        result.kept_component_count,
        count,
        removed_pixels,
        reasons or "none",
    )
    return result


def component_statistics(mask: np.ndarray) -> List[Dict[str, float]]:
    """Per-component descriptors, useful for debugging threshold choices."""
    labels, count = ndi.label(mask.astype(bool), structure=STRUCTURE_8)
    stats: List[Dict[str, float]] = []
    for index, slices in enumerate(ndi.find_objects(labels), start=1):
        if slices is None:
            continue
        sub = labels[slices] == index
        rows, cols = np.nonzero(sub)
        stats.append(
            {
                "component_id": float(index),
                "area": float(sub.sum()),
                "bbox_height": float(slices[0].stop - slices[0].start),
                "bbox_width": float(slices[1].stop - slices[1].start),
                "elongation": _elongation(rows, cols),
                "skeleton_length": float(sk_skeletonize(sub).sum()),
            }
        )
    return stats
