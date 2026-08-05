"""Probability map -> binary mask with hysteresis thresholding."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Optional

import numpy as np
from scipy import ndimage as ndi

from ..config import PreprocessingConfig

logger = logging.getLogger(__name__)

STRUCTURE_8 = np.ones((3, 3), dtype=bool)


@dataclass
class ThresholdResult:
    """Binary mask plus provenance so downstream code knows what it is looking at."""

    mask: np.ndarray
    probability_available: bool
    high_threshold: Optional[float]
    low_threshold: Optional[float]
    pixel_count: int

    def as_dict(self) -> dict:
        return {
            "probability_available": self.probability_available,
            "high_threshold": self.high_threshold,
            "low_threshold": self.low_threshold,
            "pixel_count": self.pixel_count,
        }


def hysteresis_threshold(probability: np.ndarray, high: float, low: float) -> np.ndarray:
    """Keep low-threshold pixels only when connected to a high-threshold seed."""
    if high < low:
        raise ValueError(f"high_threshold ({high}) must be >= low_threshold ({low})")
    if not 0.0 <= low <= 1.0 or not 0.0 <= high <= 1.0:
        raise ValueError("hysteresis thresholds must lie in [0, 1]")
    if probability.ndim != 2:
        raise ValueError(f"probability map must be 2D, got shape {probability.shape}")
    if not probability.size:
        return np.zeros_like(probability, dtype=bool)
    if not np.isfinite(probability).all():
        raise ValueError("probability map contains NaN or infinite values")
    strong = probability >= high
    weak = probability >= low
    if not strong.any():
        logger.warning("no pixel passes the high threshold %.3f", high)
        return np.zeros_like(strong, dtype=bool)
    labels, _ = ndi.label(weak, structure=STRUCTURE_8)
    keep = np.unique(labels[strong])
    keep = keep[keep > 0]
    return np.isin(labels, keep)


def binarize(
    probability: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    config: PreprocessingConfig,
) -> ThresholdResult:
    """Produce a boolean mask from a probability map, or fall back to a binary mask."""
    if probability is not None:
        prob = np.asarray(probability, dtype=np.float32)
        if prob.ndim != 2:
            raise ValueError(f"probability map must be 2D, got shape {prob.shape}")
        if prob.size and prob.max() > 1.5:  # tolerate 0..255 probability images
            prob = prob / 255.0
        binary = hysteresis_threshold(prob, config.high_threshold, config.low_threshold)
        return ThresholdResult(
            mask=binary,
            probability_available=True,
            high_threshold=config.high_threshold,
            low_threshold=config.low_threshold,
            pixel_count=int(binary.sum()),
        )

    if mask is None:
        raise ValueError("either a probability map or a binary mask must be provided")

    arr = np.asarray(mask)
    binary = arr > config.binary_mask_threshold if arr.dtype != bool else arr
    logger.warning("probability feature unavailable: falling back to the binary mask")
    return ThresholdResult(
        mask=np.ascontiguousarray(binary.astype(bool)),
        probability_available=False,
        high_threshold=None,
        low_threshold=None,
        pixel_count=int(binary.sum()),
    )
