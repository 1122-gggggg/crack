"""Topology-preserving skeletonization plus width estimation."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize as sk_skeletonize

logger = logging.getLogger(__name__)

NEIGHBOR_KERNEL = np.array([[1, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=np.uint8)
CONNECTIVITY_8 = np.ones((3, 3), dtype=bool)
CONNECTIVITY_4 = np.array([[0, 1, 0], [1, 1, 1], [0, 1, 0]], dtype=bool)


@dataclass
class SkeletonResult:
    """Skeleton plus the geometry needed to map edges back to line width."""

    skeleton: np.ndarray
    mask: np.ndarray
    distance: np.ndarray
    degree: np.ndarray
    component_labels: np.ndarray
    component_count: int

    @property
    def shape(self) -> Tuple[int, int]:
        return self.skeleton.shape  # type: ignore[return-value]

    def width_at(self, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
        """Local line width (in pixels) sampled at skeleton coordinates."""
        return 2.0 * self.distance[rows, cols]

    def skeleton_length(self) -> float:
        return float(self.skeleton.sum())


def neighbor_count(binary: np.ndarray) -> np.ndarray:
    """8-neighbourhood count for every foreground pixel (0 elsewhere)."""
    counts = ndi.convolve(binary.astype(np.uint8), NEIGHBOR_KERNEL, mode="constant", cval=0)
    return np.where(binary, counts, 0).astype(np.int16)


def structure_for(connectivity: int) -> np.ndarray:
    if connectivity == 8:
        return CONNECTIVITY_8
    if connectivity == 4:
        return CONNECTIVITY_4
    raise ValueError(f"connectivity must be 4 or 8, got {connectivity}")


def skeletonize_mask(
    mask: np.ndarray,
    *,
    connectivity: int = 8,
    method: str = "lee",
) -> SkeletonResult:
    """Skeletonize a binary mask without destroying the input.

    Parameters
    ----------
    mask:
        Boolean or uint8 foreground mask of the full stitched image.
    connectivity:
        Connectivity used for labelling connected components (4 or 8).
    method:
        ``skimage.morphology.skeletonize`` method; ``lee`` is more robust on
        thick blobs, ``zhang`` is faster on thin lines.
    """
    binary = np.ascontiguousarray(mask.astype(bool))
    if binary.ndim != 2:
        raise ValueError(f"mask must be 2D, got shape {binary.shape}")

    if not binary.any():
        logger.warning("empty mask received; returning empty skeleton")
        zeros_f = np.zeros(binary.shape, dtype=np.float32)
        return SkeletonResult(
            skeleton=np.zeros(binary.shape, dtype=bool),
            mask=binary,
            distance=zeros_f,
            degree=np.zeros(binary.shape, dtype=np.int16),
            component_labels=np.zeros(binary.shape, dtype=np.int32),
            component_count=0,
        )

    skeleton = sk_skeletonize(binary, method=method).astype(bool)
    distance = ndi.distance_transform_edt(binary).astype(np.float32)
    degree = neighbor_count(skeleton)
    labels, count = ndi.label(skeleton, structure=structure_for(connectivity))
    logger.info(
        "skeleton: %d px (mask %d px), %d components",
        int(skeleton.sum()),
        int(binary.sum()),
        count,
    )
    return SkeletonResult(
        skeleton=skeleton,
        mask=binary,
        distance=distance,
        degree=degree,
        component_labels=labels.astype(np.int32),
        component_count=int(count),
    )


def skeleton_debug_image(
    result: SkeletonResult,
    image: Optional[np.ndarray] = None,
    *,
    skeleton_color: Tuple[int, int, int] = (0, 255, 255),
    mask_color: Tuple[int, int, int] = (90, 90, 90),
) -> np.ndarray:
    """Render skeleton over mask (or over the RGB image when provided)."""
    if image is None:
        canvas = np.zeros((*result.shape, 3), dtype=np.uint8)
        canvas[result.mask] = mask_color
    else:
        canvas = image.copy()
        if canvas.ndim == 2:
            canvas = np.dstack([canvas] * 3)
    canvas[result.skeleton] = skeleton_color
    return canvas
