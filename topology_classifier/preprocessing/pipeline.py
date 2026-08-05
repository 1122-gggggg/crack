"""Ordered preprocessing pipeline: threshold -> cleanup -> optional gap repair."""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ..config import PreprocessingConfig
from .cleanup import CleanupResult, remove_noise_components
from .gap_repair import GapRepairResult, gap_repair_debug_image, repair_gaps
from .hysteresis import ThresholdResult, binarize

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedMask:
    """Final boolean mask plus a full record of what each stage did."""

    mask: np.ndarray
    raw_mask: np.ndarray
    threshold: ThresholdResult
    cleanup: CleanupResult
    gap_repair: GapRepairResult
    stats: Dict[str, object] = field(default_factory=dict)

    @property
    def probability_available(self) -> bool:
        return self.threshold.probability_available

    def as_dict(self) -> Dict[str, object]:
        return {
            "threshold": self.threshold.as_dict(),
            "cleanup": self.cleanup.as_dict(),
            "gap_repair": self.gap_repair.as_dict(),
            "final_pixel_count": int(self.mask.sum()),
            **self.stats,
        }


def preprocess_mask(
    config: PreprocessingConfig,
    probability: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    debug_dir: Optional[Path] = None,
    image_id: str = "image",
) -> PreprocessedMask:
    """Run the full preprocessing chain on one image.

    Args:
        config: Preprocessing thresholds.
        probability: ``float32`` probability map in ``[0, 1]``, if available.
        mask: Binary fallback used only when ``probability`` is ``None``.
        debug_dir: When set (and ``save_debug_masks`` is on), debug PNGs go here.
        image_id: Identifier used for debug filenames and log messages.

    Returns:
        A :class:`PreprocessedMask` carrying the mask and per-stage statistics.
    """
    threshold = binarize(probability, mask, config)
    raw_mask = threshold.mask.copy()

    cleanup = remove_noise_components(threshold.mask, config, probability=probability)
    gap = repair_gaps(cleanup.mask, config, probability=probability)

    final_mask = np.ascontiguousarray(gap.mask.astype(bool))
    result = PreprocessedMask(
        mask=final_mask,
        raw_mask=raw_mask,
        threshold=threshold,
        cleanup=cleanup,
        gap_repair=gap,
        stats={"image_id": image_id, "height": int(final_mask.shape[0]), "width": int(final_mask.shape[1])},
    )

    if debug_dir is not None and config.save_debug_masks:
        _write_debug(debug_dir, image_id, raw_mask, cleanup.mask, final_mask, gap)

    logger.info(
        "%s: %d px after threshold -> %d px after cleanup -> %d px final",
        image_id,
        int(raw_mask.sum()),
        int(cleanup.mask.sum()),
        int(final_mask.sum()),
    )
    return result


def _write_debug(
    debug_dir: Path,
    image_id: str,
    raw_mask: np.ndarray,
    cleaned: np.ndarray,
    final_mask: np.ndarray,
    gap: GapRepairResult,
) -> None:
    import cv2

    debug_dir.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(debug_dir / f"{image_id}_01_threshold.png"), raw_mask.astype(np.uint8) * 255)
    cv2.imwrite(str(debug_dir / f"{image_id}_02_cleaned.png"), cleaned.astype(np.uint8) * 255)
    if gap.enabled:
        cv2.imwrite(str(debug_dir / f"{image_id}_03_gap_repair.png"), gap_repair_debug_image(cleaned, final_mask))
