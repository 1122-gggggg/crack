"""Shared label vocabulary for edge-level supervision."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

from ..config import ClassesConfig

UNLABELED = "unlabeled"
UNCERTAIN = "uncertain"


@dataclass(frozen=True)
class EdgeLabel:
    """Label assigned to one edge, with the evidence that produced it."""

    edge_id: int
    label: str
    purity: float
    pixel_count: int
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "edge_id": self.edge_id,
            "label": self.label,
            "label_purity": round(self.purity, 4),
            "label_pixel_count": self.pixel_count,
            "label_reason": self.reason,
        }


class LabelVocabulary:
    """Maps between class names, pixel values and training indices.

    ``uncertain`` and ``unlabeled`` are deliberately *not* training classes: a
    mixed edge must never be forced into ``crack`` or ``craquelure``.
    """

    def __init__(self, classes: ClassesConfig) -> None:
        self.classes = classes
        self.names: List[str] = classes.active_names()
        self._name_to_index = {name: index for index, name in enumerate(self.names)}
        self._value_to_name = {classes.pixel_value(name): name for name in self.names}

    @property
    def background_value(self) -> int:
        return self.classes.background

    @property
    def ignore_value(self) -> int:
        return self.classes.ignore

    def is_trainable(self, label: str) -> bool:
        return label in self._name_to_index

    def index_of(self, label: str) -> Optional[int]:
        return self._name_to_index.get(label)

    def name_of_value(self, value: int) -> Optional[str]:
        return self._value_to_name.get(int(value))

    def value_of_name(self, name: str) -> int:
        if name in (UNCERTAIN, UNLABELED):
            return self.ignore_value
        if name not in self._name_to_index:
            raise KeyError(f"unknown class name: {name!r} (known: {self.names})")
        return self.classes.pixel_value(name)

    def label_values(self) -> Dict[str, int]:
        return {name: self.classes.pixel_value(name) for name in self.names}
