"""Grouped cross-validation and final fit for the handcrafted-feature baseline."""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..config import TopologyConfig
from ..logging_utils import environment_report, write_json
from ..models.baseline import BaselineClassifier
from ..visualization.reports import confusion_matrix_figure, feature_importance_figure, write_report
from .dataset import feature_matrix, numeric_feature_columns, prepare_training_frame
from .metrics import EvaluationResult, aggregate_folds, evaluate
from .splits import SplitReport, describe_groups, split_frame

logger = logging.getLogger(__name__)


def set_seed(seed: int) -> None:
    """Deterministic seeds for every RNG the baseline path touches."""
    random.seed(seed)
    np.random.seed(seed)


@dataclass
class TrainingOutcome:
    """Everything produced by a training run."""

    fold_results: List[EvaluationResult]
    aggregate: Dict[str, object]
    split_report: SplitReport
    class_names: List[str]
    feature_columns: List[str]
    model_path: Optional[Path] = None
    report_path: Optional[Path] = None
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, object]:
        return {
            "aggregate": self.aggregate,
            "split": self.split_report.as_dict(),
            "class_names": self.class_names,
            "feature_count": len(self.feature_columns),
            "folds": [r.as_dict() for r in self.fold_results],
            "model_path": str(self.model_path) if self.model_path else None,
            "notes": self.notes,
        }


def train_baseline(
    config: TopologyConfig,
    features: pd.DataFrame,
    output_dir: Optional[Path] = None,
) -> TrainingOutcome:
    """Cross-validate, then refit on all labelled data and save the model.

    Raises:
        ValueError: If there are no trainable rows, or fewer than two groups.
    """
    set_seed(config.model.random_seed)
    output_dir = Path(output_dir) if output_dir else config.output_dir / "baseline"
    output_dir.mkdir(parents=True, exist_ok=True)

    frame, class_names = prepare_training_frame(features, config)
    feature_columns = numeric_feature_columns(frame)
    if not feature_columns:
        raise ValueError("no numeric feature columns found in the feature table")

    group_column = config.training.group_column
    if group_column not in frame.columns:
        raise KeyError(f"group column {group_column!r} not in the feature table")

    notes: List[str] = []
    group_table = describe_groups(frame, group_column, "label")
    logger.info("group/class table:\n%s", group_table)

    folds, split_report = split_frame(
        frame,
        label_column="y",
        group_column=group_column,
        n_splits=config.training.n_splits,
        seed=config.model.random_seed,
    )
    notes.extend(split_report.warnings)

    X = feature_matrix(frame, feature_columns)
    y = frame["y"].to_numpy(dtype=int)
    lengths = frame["length_px"].to_numpy(dtype=float) if "length_px" in frame.columns else None

    fold_results: List[EvaluationResult] = []
    for index, (train_index, test_index) in enumerate(folds):
        model = BaselineClassifier(
            config.model,
            config.training,
            feature_names=feature_columns,
            class_names=class_names,
            config_hash=config.config_hash(scope="features"),
        )
        try:
            model.fit(X[train_index], y[train_index])
        except ValueError as error:
            message = f"fold {index}: skipped ({error})"
            logger.warning(message)
            notes.append(message)
            continue
        probabilities = model.predict_proba(X[test_index])
        predictions = np.argmax(probabilities, axis=1)
        confidence = probabilities.max(axis=1)
        abstained = confidence < config.model.minimum_prediction_confidence
        result = evaluate(
            y[test_index],
            predictions,
            class_names,
            lengths=lengths[test_index] if lengths is not None else None,
            abstained=abstained,
        )
        logger.info("fold %d: %s", index, result.summary_line())
        fold_results.append(result)

    if not fold_results:
        raise ValueError("every fold was skipped; not enough labelled data to evaluate honestly")

    aggregate = aggregate_folds(fold_results)

    final_model = BaselineClassifier(
        config.model,
        config.training,
        feature_names=feature_columns,
        class_names=class_names,
        config_hash=config.config_hash(scope="features"),
    ).fit(X, y)
    model_path = final_model.save(output_dir / "baseline_model.joblib")

    importances = final_model.feature_importances()
    write_json(output_dir / "feature_importances.json", importances)
    feature_importance_figure(importances, output_dir / "feature_importances.png")
    confusion_matrix_figure(
        np.asarray(aggregate["confusion_sum"], dtype=int),
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"{final_model.backend} - {split_report.strategy}",
    )

    report_path = write_report(
        output_dir / "baseline_report.md",
        title="Baseline edge classifier",
        sections={
            "Data": {
                "labelled_edges": len(frame),
                "feature_count": len(feature_columns),
                "class_counts": frame["label"].value_counts().to_dict(),
                "group_column": group_column,
                "group_count": split_report.group_count,
            },
            "Split": split_report.as_dict(),
            "Cross-validated metrics": {k: v for k, v in aggregate.items() if k != "confusion_sum"},
            "Per-fold": [r.summary_line() for r in fold_results],
            "Notes": notes or ["none"],
            "Environment": environment_report(),
        },
    )

    outcome = TrainingOutcome(
        fold_results=fold_results,
        aggregate=aggregate,
        split_report=split_report,
        class_names=class_names,
        feature_columns=feature_columns,
        model_path=model_path,
        report_path=report_path,
        notes=notes,
    )
    write_json(output_dir / "training_outcome.json", outcome.as_dict())
    return outcome
