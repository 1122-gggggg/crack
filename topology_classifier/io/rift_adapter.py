"""Read Stage-1 RIFT outputs without modifying RIFT itself.

RIFT (``infer_tiles.py``) writes, per image:

* ``<stem>_prob.npy``  -- ``float32`` probability map, ``H x W``, range ``0..1``
* ``<stem>_mask.npy``  -- ``uint8`` binary mask, ``H x W``, values ``0/1``
* ``<stem>_prob.png`` / ``<stem>_mask.png`` -- 8-bit previews

Filenames may carry an inference tag (scale, checkpoint, stride, TTA), e.g.
``KJTHT-SC-L-1RB1-1_s1_TUT_base_st256_prob.npy``. The adapter therefore matches
by prefix and prefers the float ``.npy`` map; when only a binary mask exists it
still runs but flags ``probability_available = False`` so that probability-based
features are reported as unavailable instead of silently faked.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger(__name__)

PROBABILITY_SUFFIXES: Tuple[str, ...] = ("_prob", "_probability", "_pred")
MASK_SUFFIXES: Tuple[str, ...] = ("_mask", "_binary", "_seg")


class MissingRiftOutputError(FileNotFoundError):
    """Raised when neither a probability map nor a binary mask can be found."""


@dataclass
class RiftOutputs:
    """One image worth of Stage-1 output."""

    image_id: str
    probability: Optional[np.ndarray]
    mask: Optional[np.ndarray]
    probability_path: Optional[Path]
    mask_path: Optional[Path]

    @property
    def probability_available(self) -> bool:
        return self.probability is not None

    @property
    def shape(self) -> Tuple[int, int]:
        array = self.probability if self.probability is not None else self.mask
        if array is None:
            raise MissingRiftOutputError(f"{self.image_id}: no RIFT output loaded")
        return int(array.shape[0]), int(array.shape[1])


def _candidates(directory: Path, image_id: str, suffixes: Sequence[str]) -> List[Path]:
    """Ranked candidate files: exact ``.npy`` first, then tagged, then ``.png``."""
    if not directory.is_dir():
        return []
    ranked: List[Tuple[int, Path]] = []
    for path in sorted(directory.iterdir()):
        if not path.is_file() or path.suffix.lower() not in {".npy", ".png", ".tif", ".tiff"}:
            continue
        stem = path.stem
        if stem != image_id and not stem.startswith(f"{image_id}_"):
            continue
        matched = any(stem.endswith(suffix) for suffix in suffixes)
        exact = stem in {f"{image_id}{suffix}" for suffix in suffixes}
        is_npy = path.suffix.lower() == ".npy"
        if not matched and stem != image_id:
            continue
        score = (0 if is_npy else 2) + (0 if exact else 1)
        ranked.append((score, path))
    ranked.sort(key=lambda item: (item[0], str(item[1])))
    return [path for _, path in ranked]


def _load_array(path: Path) -> np.ndarray:
    if path.suffix.lower() == ".npy":
        return np.load(path)
    import cv2

    array = cv2.imread(str(path), cv2.IMREAD_UNCHANGED)
    if array is None:
        raise OSError(f"failed to read {path}")
    if array.ndim == 3:
        array = array[..., 0]
    return array


def _as_probability(array: np.ndarray, path: Path) -> np.ndarray:
    prob = np.asarray(array, dtype=np.float32)
    if prob.ndim != 2:
        raise ValueError(f"{path}: expected a 2D probability map, got shape {prob.shape}")
    if not prob.size:
        raise ValueError(f"{path}: probability map is empty")
    if not np.isfinite(prob).all():
        raise ValueError(f"{path}: probability map contains NaN or infinite values")
    maximum = float(prob.max())
    if maximum > 1.5:
        prob = prob / 255.0
        logger.debug("%s: rescaled 0..255 probability image to 0..1", path.name)
    return np.clip(prob, 0.0, 1.0)


def _as_mask(array: np.ndarray, path: Path, threshold: int) -> np.ndarray:
    if array.ndim != 2:
        raise ValueError(f"{path}: expected a 2D mask, got shape {array.shape}")
    if array.dtype == bool:
        return array
    maximum = int(array.max()) if array.size else 0
    return array > (threshold if maximum > 1 else 0)


class RiftAdapter:
    """Locate and load RIFT probability maps and masks for a given image id."""

    def __init__(
        self,
        probability_dir: Path | str,
        mask_dir: Optional[Path | str] = None,
        binary_mask_threshold: int = 127,
    ) -> None:
        self.probability_dir = Path(probability_dir)
        self.mask_dir = Path(mask_dir) if mask_dir is not None else self.probability_dir
        self.binary_mask_threshold = binary_mask_threshold

    def find_probability(self, image_id: str) -> Optional[Path]:
        for path in _candidates(self.probability_dir, image_id, PROBABILITY_SUFFIXES):
            return path
        return None

    def find_mask(self, image_id: str) -> Optional[Path]:
        for path in _candidates(self.mask_dir, image_id, MASK_SUFFIXES):
            return path
        return None

    def load(self, image_id: str) -> RiftOutputs:
        """Load Stage-1 outputs, preferring the float probability map.

        Raises:
            MissingRiftOutputError: If neither a probability map nor a mask exists.
            ValueError: If the arrays are not 2D or their shapes disagree.
        """
        probability_path = self.find_probability(image_id)
        mask_path = self.find_mask(image_id)

        probability: Optional[np.ndarray] = None
        mask: Optional[np.ndarray] = None

        if probability_path is not None:
            probability = _as_probability(_load_array(probability_path), probability_path)
        if mask_path is not None:
            mask = _as_mask(_load_array(mask_path), mask_path, self.binary_mask_threshold)

        if probability is None and mask is None:
            raise MissingRiftOutputError(
                f"{image_id}: no RIFT output under {self.probability_dir} or {self.mask_dir}"
            )
        if probability is None:
            logger.warning(
                "%s: probability feature unavailable, using binary mask %s",
                image_id,
                mask_path.name if mask_path else "?",
            )
        if probability is not None and mask is not None and probability.shape != mask.shape:
            raise ValueError(
                f"{image_id}: probability shape {probability.shape} != mask shape {mask.shape}"
            )

        return RiftOutputs(
            image_id=image_id,
            probability=probability,
            mask=mask,
            probability_path=probability_path,
            mask_path=mask_path,
        )
