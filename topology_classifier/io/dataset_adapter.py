"""Enumerate the images to process and attach their grouping metadata."""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from ..config import DataConfig, TopologyConfig
from .rift_adapter import MissingRiftOutputError, RiftAdapter, RiftOutputs

logger = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


@dataclass(frozen=True)
class ImageRecord:
    """One processing unit: an image id, its optional RGB file and its group."""

    image_id: str
    panel_id: str
    image_path: Optional[Path] = None
    probability_path: Optional[Path] = None
    mask_path: Optional[Path] = None
    label_path: Optional[Path] = None

    def as_dict(self) -> Dict[str, Optional[str]]:
        return {
            "image_id": self.image_id,
            "panel_id": self.panel_id,
            "image_path": str(self.image_path) if self.image_path else None,
            "probability_path": str(self.probability_path) if self.probability_path else None,
            "mask_path": str(self.mask_path) if self.mask_path else None,
            "label_path": str(self.label_path) if self.label_path else None,
        }


def _strip_inference_tag(stem: str, known_ids: Sequence[str]) -> str:
    """Recover the image id from a tagged RIFT filename."""
    matches = [
        image_id
        for image_id in known_ids
        if stem == image_id or stem.startswith(f"{image_id}_")
    ]
    if matches:
        return max(matches, key=len)
    for suffix in ("_prob", "_probability", "_pred", "_mask", "_binary", "_seg"):
        if stem.endswith(suffix):
            return stem[: -len(suffix)]
    return stem


def _read_panel_mapping(path: Path) -> Dict[str, str]:
    mapping: Dict[str, str] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or "image_id" not in reader.fieldnames:
            raise ValueError(f"{path}: panel mapping needs an 'image_id' column")
        if "panel_id" not in reader.fieldnames:
            raise ValueError(f"{path}: panel mapping needs a 'panel_id' column")
        for row in reader:
            image_id = (row.get("image_id") or "").strip()
            panel_id = (row.get("panel_id") or "").strip()
            if image_id and panel_id:
                mapping[image_id] = panel_id
    logger.info("loaded panel mapping for %d images from %s", len(mapping), path.name)
    return mapping


class DatasetAdapter:
    """Discovers images and pairs them with their Stage-1 outputs and labels."""

    def __init__(self, config: TopologyConfig) -> None:
        self.config = config
        self.data: DataConfig = config.data
        self.rift = RiftAdapter(
            probability_dir=self.data.rift_prob_dir,
            mask_dir=self.data.rift_mask_dir,
            binary_mask_threshold=config.preprocessing.binary_mask_threshold,
        )
        self._panel_mapping = self._load_panel_mapping()

    def _load_panel_mapping(self) -> Dict[str, str]:
        if not self.data.panel_mapping_csv:
            return {}
        path = Path(self.data.panel_mapping_csv)
        if not path.is_file():
            raise FileNotFoundError(f"panel_mapping_csv not found: {path}")
        return _read_panel_mapping(path)

    def panel_id_for(self, image_id: str) -> str:
        """Group id used by the grouped cross-validation split."""
        if image_id in self._panel_mapping:
            return self._panel_mapping[image_id]
        return image_id

    def _image_files(self) -> Dict[str, Path]:
        images_dir = Path(self.data.images_dir)
        found: Dict[str, Path] = {}
        if not images_dir.is_dir():
            logger.warning("images_dir %s does not exist; running without RGB images", images_dir)
            return found
        patterns = [p.strip() for p in self.data.image_glob.split(",") if p.strip()]
        for pattern in patterns:
            for path in sorted(images_dir.glob(pattern)):
                if path.suffix.lower() in IMAGE_EXTENSIONS:
                    found.setdefault(path.stem, path)
        return found

    def _ids_from_rift_dir(self, known_ids: Sequence[str]) -> List[str]:
        ids: List[str] = []
        for directory in {Path(self.data.rift_prob_dir), Path(self.data.rift_mask_dir)}:
            if not directory.is_dir():
                continue
            for path in sorted(directory.iterdir()):
                if path.suffix.lower() not in {".npy", ".png", ".tif", ".tiff"}:
                    continue
                ids.append(_strip_inference_tag(path.stem, known_ids))
        return sorted(set(ids))

    def records(self) -> List[ImageRecord]:
        """All discoverable images, ordered by image id.

        Image ids come from ``images_dir`` when it exists; otherwise they are
        recovered from the RIFT output filenames so the pipeline can run on
        Stage-1 artefacts alone.
        """
        image_files = self._image_files()
        image_ids = sorted(image_files) if image_files else self._ids_from_rift_dir([])
        if not image_ids:
            logger.warning(
                "no images discovered under %s or %s",
                self.data.images_dir,
                self.data.rift_prob_dir,
            )

        labels_dir = Path(self.data.labels_dir)
        records: List[ImageRecord] = []
        for image_id in image_ids:
            label_path: Optional[Path] = None
            if labels_dir.is_dir():
                for extension in (".png", ".tif", ".tiff", ".npy"):
                    candidate = labels_dir / f"{image_id}{extension}"
                    if candidate.is_file():
                        label_path = candidate
                        break
            records.append(
                ImageRecord(
                    image_id=image_id,
                    panel_id=self.panel_id_for(image_id),
                    image_path=image_files.get(image_id),
                    probability_path=self.rift.find_probability(image_id),
                    mask_path=self.rift.find_mask(image_id),
                    label_path=label_path,
                )
            )

        if not self._panel_mapping and records:
            logger.warning(
                "no panel_mapping_csv:每張影像自成一組 (panel_id = image_id); "
                "grouped cross-validation cannot protect against same-panel leakage"
            )
        logger.info("dataset: %d image(s) discovered", len(records))
        return records

    def iter_records(self) -> Iterator[ImageRecord]:
        yield from self.records()

    def load_rift(self, record: ImageRecord) -> RiftOutputs:
        """Load Stage-1 arrays for one record.

        Raises:
            MissingRiftOutputError: If the record has no usable Stage-1 output.
        """
        return self.rift.load(record.image_id)

    def load_image(self, record: ImageRecord) -> Optional[np.ndarray]:
        """Load the RGB image as BGR ``uint8``, or ``None`` when unavailable."""
        if record.image_path is None:
            return None
        import cv2

        image = cv2.imread(str(record.image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise OSError(f"failed to read image {record.image_path}")
        return image


__all__ = ["DatasetAdapter", "ImageRecord", "MissingRiftOutputError"]
