"""Evaluation metrics for edge classification.

Edge counts alone are misleading: a panel can contain thousands of tiny
craquelure edges and a handful of long cracks. Every metric is therefore
reported twice -- unweighted, and weighted by skeleton length in pixels.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_fscore_support,
)

logger = logging.getLogger(__name__)

NAN = float("nan")


@dataclass
class EvaluationResult:
    """Full metric bundle for one evaluation run."""

    class_names: List[str]
    support: Dict[str, int]
    accuracy: float
    balanced_accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: Dict[str, Dict[str, float]]
    confusion: np.ndarray
    length_weighted: Dict[str, float] = field(default_factory=dict)
    abstention_rate: float = 0.0
    coverage: float = 1.0
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "class_names": self.class_names,
            "support": self.support,
            "accuracy": self.accuracy,
            "balanced_accuracy": self.balanced_accuracy,
            "macro_f1": self.macro_f1,
            "weighted_f1": self.weighted_f1,
            "per_class": self.per_class,
            "confusion": self.confusion.tolist(),
            "length_weighted": self.length_weighted,
            "abstention_rate": self.abstention_rate,
            "coverage": self.coverage,
            "notes": self.notes,
        }

    def summary_line(self) -> str:
        return (
            f"n={sum(self.support.values())} acc={self.accuracy:.4f} "
            f"balanced_acc={self.balanced_accuracy:.4f} macro_f1={self.macro_f1:.4f}"
        )


def evaluate(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    class_names: Sequence[str],
    lengths: Optional[Sequence[float]] = None,
    abstained: Optional[Sequence[bool]] = None,
) -> EvaluationResult:
    """Compute the metric bundle.

    Args:
        y_true: Ground-truth class indices.
        y_pred: Predicted class indices, aligned with ``y_true``.
        class_names: Index-ordered class names.
        lengths: Per-edge skeleton length used for length-weighted metrics.
        abstained: Per-edge flag marking predictions below the confidence
            threshold; those rows are excluded from accuracy and counted in
            ``abstention_rate`` instead of being scored as errors.

    Raises:
        ValueError: If the inputs have mismatched lengths.
    """
    true_array = np.asarray(y_true, dtype=int)
    pred_array = np.asarray(y_pred, dtype=int)
    if true_array.shape != pred_array.shape:
        raise ValueError(f"y_true {true_array.shape} and y_pred {pred_array.shape} differ")
    if true_array.size == 0:
        return EvaluationResult(
            class_names=list(class_names),
            support={},
            accuracy=NAN,
            balanced_accuracy=NAN,
            macro_f1=NAN,
            weighted_f1=NAN,
            per_class={},
            confusion=np.zeros((len(class_names), len(class_names)), dtype=int),
            notes=["empty evaluation set"],
        )

    notes: List[str] = []
    abstain_array = (
        np.asarray(abstained, dtype=bool) if abstained is not None else np.zeros_like(true_array, dtype=bool)
    )
    abstention_rate = float(abstain_array.mean())
    scored = ~abstain_array
    if not scored.any():
        notes.append("every prediction abstained; no scored rows")
        return EvaluationResult(
            class_names=list(class_names),
            support={},
            accuracy=NAN,
            balanced_accuracy=NAN,
            macro_f1=NAN,
            weighted_f1=NAN,
            per_class={},
            confusion=np.zeros((len(class_names), len(class_names)), dtype=int),
            abstention_rate=abstention_rate,
            coverage=0.0,
            notes=notes,
        )

    true_scored = true_array[scored]
    pred_scored = pred_array[scored]
    indices = list(range(len(class_names)))

    precision, recall, f1, support = precision_recall_fscore_support(
        true_scored, pred_scored, labels=indices, zero_division=0
    )
    per_class = {
        name: {
            "precision": float(precision[i]),
            "recall": float(recall[i]),
            "f1": float(f1[i]),
            "support": int(support[i]),
        }
        for i, name in enumerate(class_names)
    }
    missing = [name for name, stats in per_class.items() if stats["support"] == 0]
    if missing:
        notes.append(f"no ground-truth samples for class(es): {missing}")

    if np.unique(true_scored).size < 2:
        # sklearn's balanced_accuracy_score emits a warning for a one-class
        # validation fold.  Its defined behaviour is simply the recall of the
        # class that is present, which we can compute without the warning.
        present_classes = np.unique(true_scored)
        balanced_accuracy = float(
            np.mean([np.mean(pred_scored[true_scored == index] == index) for index in present_classes])
        )
        notes.append("validation rows contain a single ground-truth class")
    else:
        balanced_accuracy = float(balanced_accuracy_score(true_scored, pred_scored))

    result = EvaluationResult(
        class_names=list(class_names),
        support={name: int(support[i]) for i, name in enumerate(class_names)},
        accuracy=float((true_scored == pred_scored).mean()),
        balanced_accuracy=balanced_accuracy,
        macro_f1=float(f1_score(true_scored, pred_scored, labels=indices, average="macro", zero_division=0)),
        weighted_f1=float(f1_score(true_scored, pred_scored, labels=indices, average="weighted", zero_division=0)),
        per_class=per_class,
        confusion=confusion_matrix(true_scored, pred_scored, labels=indices),
        abstention_rate=abstention_rate,
        coverage=float(scored.mean()),
        notes=notes,
    )

    if lengths is not None:
        weights = np.asarray(lengths, dtype=np.float64)[scored]
        if weights.size and weights.sum() > 0:
            correct = (true_scored == pred_scored).astype(np.float64)
            result.length_weighted = {
                "accuracy": float(np.sum(correct * weights) / np.sum(weights)),
                "total_length_px": float(weights.sum()),
                **{
                    f"recall_{name}": _length_recall(true_scored, pred_scored, weights, index)
                    for index, name in enumerate(class_names)
                },
            }
    return result


def _length_recall(y_true: np.ndarray, y_pred: np.ndarray, weights: np.ndarray, index: int) -> float:
    selected = y_true == index
    total = float(weights[selected].sum())
    if total <= 0:
        return NAN
    hit = float(weights[selected & (y_pred == index)].sum())
    return hit / total


def aggregate_folds(results: Sequence[EvaluationResult]) -> Dict[str, object]:
    """Mean and standard deviation of headline metrics across folds."""
    if not results:
        return {"folds": 0}
    keys = ("accuracy", "balanced_accuracy", "macro_f1", "weighted_f1")
    payload: Dict[str, object] = {"folds": len(results)}
    for key in keys:
        values = np.asarray([getattr(r, key) for r in results], dtype=np.float64)
        finite = values[~np.isnan(values)]
        payload[f"{key}_mean"] = float(finite.mean()) if finite.size else NAN
        payload[f"{key}_std"] = float(finite.std()) if finite.size else NAN
    total = np.sum([r.confusion for r in results], axis=0)
    payload["confusion_sum"] = total.tolist()
    payload["class_names"] = results[0].class_names
    return payload
