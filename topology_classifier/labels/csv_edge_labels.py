"""Read human edge annotations produced by ``export-edge-review``.

The reviewer fills in the ``label`` column of the exported CSV. Nothing here
invents labels: blank rows stay ``unlabeled`` and are dropped from training,
and any value outside the configured vocabulary is a hard error rather than a
silent remap.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, Iterable, List

import pandas as pd

from ..config import ClassesConfig
from .label_types import UNCERTAIN, UNLABELED, LabelVocabulary

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = ("image_id", "edge_id", "label")
REVIEW_COLUMNS = (
    "image_id",
    "panel_id",
    "component_id",
    "edge_id",
    "label",
    "confidence",
    "notes",
)
ALIASES: Dict[str, str] = {
    "": UNLABELED,
    "nan": UNLABELED,
    "none": UNLABELED,
    "unknown": UNCERTAIN,
    "uncertain": UNCERTAIN,
    "ignore": UNCERTAIN,
    "mixed": UNCERTAIN,
    "c": "crack",
    "crack": "crack",
    "q": "craquelure",
    "craquelure": "craquelure",
    "craqueleur": "craquelure",
    "o": "other_line",
    "other": "other_line",
    "other_line": "other_line",
}


def normalize_label(raw: object, vocabulary: LabelVocabulary) -> str:
    """Map one raw cell to a canonical label name.

    Raises:
        ValueError: If the value is neither a known alias, a class name nor a
            configured class pixel value.
    """
    text = str(raw).strip().lower() if raw is not None else ""
    if text in ALIASES:
        candidate = ALIASES[text]
    elif text in {name.lower() for name in vocabulary.names}:
        candidate = text
    elif text.isdigit():
        name = vocabulary.name_of_value(int(text))
        if name is None and int(text) == vocabulary.ignore_value:
            return UNCERTAIN
        if name is None and int(text) == vocabulary.background_value:
            return UNLABELED
        if name is None:
            raise ValueError(f"unknown label value {text!r}")
        candidate = name
    else:
        raise ValueError(f"unknown label {raw!r}; expected one of {sorted(set(ALIASES) | set(vocabulary.names))}")

    if candidate in (UNCERTAIN, UNLABELED):
        return candidate
    if not vocabulary.is_trainable(candidate):
        raise ValueError(
            f"label {candidate!r} is not enabled in the config (active classes: {vocabulary.names})"
        )
    return candidate


def read_edge_annotations(path: Path | str, classes: ClassesConfig) -> pd.DataFrame:
    """Load and validate a reviewed edge-label CSV.

    Returns:
        DataFrame with ``image_id``, ``edge_id``, canonical ``label`` and any
        extra review columns that were present.

    Raises:
        FileNotFoundError: If the CSV does not exist.
        ValueError: On missing columns, duplicate keys or unknown label values.
    """
    csv_path = Path(path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"edge annotation CSV not found: {csv_path}")

    frame = pd.read_csv(csv_path, dtype={"image_id": str, "panel_id": str})
    missing = [column for column in REQUIRED_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"{csv_path}: missing required column(s) {missing}")

    vocabulary = LabelVocabulary(classes)
    errors: List[str] = []
    normalized: List[str] = []
    for index, raw in enumerate(frame["label"].tolist()):
        try:
            normalized.append(normalize_label(raw, vocabulary))
        except ValueError as error:
            errors.append(f"row {index + 2}: {error}")
            normalized.append(UNLABELED)
    if errors:
        raise ValueError(f"{csv_path}: {len(errors)} invalid label(s):\n  " + "\n  ".join(errors[:20]))

    frame["label"] = normalized
    frame["edge_id"] = frame["edge_id"].astype(int)

    duplicates = frame.duplicated(subset=["image_id", "edge_id"], keep=False)
    if duplicates.any():
        offending = frame.loc[duplicates, ["image_id", "edge_id"]].to_dict("records")[:10]
        raise ValueError(f"{csv_path}: duplicate (image_id, edge_id) rows, e.g. {offending}")

    counts = frame["label"].value_counts().to_dict()
    logger.info("%s: %d annotated edges %s", csv_path.name, len(frame), counts)
    return frame


def trainable_subset(frame: pd.DataFrame, classes: ClassesConfig) -> pd.DataFrame:
    """Keep only rows whose label is an actual training class."""
    vocabulary = LabelVocabulary(classes)
    keep = frame["label"].map(vocabulary.is_trainable)
    dropped = int((~keep).sum())
    if dropped:
        logger.info("dropping %d non-trainable row(s) (uncertain/unlabeled)", dropped)
    return frame.loc[keep].reset_index(drop=True)


def label_distribution(labels: Iterable[str]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for label in labels:
        counts[label] = counts.get(label, 0) + 1
    return dict(sorted(counts.items()))
