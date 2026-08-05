"""Gradient-boosted baseline over the handcrafted edge features.

XGBoost is preferred when installed; otherwise the class falls back to
scikit-learn's ``HistGradientBoostingClassifier`` or a random forest. The active
backend is recorded in the model metadata so a reported score can always be
traced back to what actually ran.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..config import ModelConfig, TrainingConfig

logger = logging.getLogger(__name__)

BACKENDS = ("xgboost", "hist_gradient_boosting", "random_forest")


@dataclass
class BaselineMetadata:
    backend: str
    feature_names: List[str]
    class_names: List[str]
    config_hash: str
    random_seed: int
    training_rows: int = 0
    class_counts: Dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, object]:
        return {
            "backend": self.backend,
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "config_hash": self.config_hash,
            "random_seed": self.random_seed,
            "training_rows": self.training_rows,
            "class_counts": self.class_counts,
        }


def _resolve_backend(requested: str) -> str:
    if requested not in BACKENDS:
        raise ValueError(f"unknown baseline_type {requested!r}; expected one of {BACKENDS}")
    if requested == "xgboost":
        try:
            import xgboost  # noqa: F401
        except ImportError:
            logger.warning("xgboost not installed; falling back to hist_gradient_boosting")
            return "hist_gradient_boosting"
    return requested


def _sample_weights(y: np.ndarray, class_count: int, weighting: str) -> Optional[np.ndarray]:
    if weighting != "balanced":
        return None
    counts = np.bincount(y, minlength=class_count).astype(np.float64)
    counts[counts == 0] = 1.0
    weights = len(y) / (class_count * counts)
    return weights[y]


class BaselineClassifier:
    """Thin wrapper giving all three backends one interface and one metadata file."""

    def __init__(
        self,
        model_config: ModelConfig,
        training_config: TrainingConfig,
        feature_names: Sequence[str],
        class_names: Sequence[str],
        config_hash: str = "",
    ) -> None:
        self.model_config = model_config
        self.training_config = training_config
        self.backend = _resolve_backend(model_config.baseline_type)
        self.metadata = BaselineMetadata(
            backend=self.backend,
            feature_names=list(feature_names),
            class_names=list(class_names),
            config_hash=config_hash,
            random_seed=model_config.random_seed,
        )
        self.model = self._build()

    def _build(self):
        seed = self.model_config.random_seed
        cfg = self.training_config
        if self.backend == "xgboost":
            from xgboost import XGBClassifier

            return XGBClassifier(
                n_estimators=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                subsample=0.9,
                colsample_bytree=0.9,
                reg_lambda=1.0,
                objective="multi:softprob" if len(self.metadata.class_names) > 2 else "binary:logistic",
                num_class=len(self.metadata.class_names) if len(self.metadata.class_names) > 2 else None,
                tree_method="hist",
                random_state=seed,
                n_jobs=0,
                eval_metric="mlogloss" if len(self.metadata.class_names) > 2 else "logloss",
            )
        if self.backend == "hist_gradient_boosting":
            from sklearn.ensemble import HistGradientBoostingClassifier

            return HistGradientBoostingClassifier(
                max_iter=cfg.n_estimators,
                max_depth=cfg.max_depth,
                learning_rate=cfg.learning_rate,
                random_state=seed,
            )
        from sklearn.ensemble import RandomForestClassifier

        return RandomForestClassifier(
            n_estimators=cfg.n_estimators,
            max_depth=cfg.max_depth or None,
            random_state=seed,
            n_jobs=-1,
            class_weight="balanced" if cfg.class_weighting == "balanced" else None,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> "BaselineClassifier":
        """Fit the model, applying balanced sample weights when configured."""
        if X.shape[0] != y.shape[0]:
            raise ValueError(f"X has {X.shape[0]} rows but y has {y.shape[0]}")
        if X.shape[0] == 0:
            raise ValueError("cannot fit on an empty training set")
        present = np.unique(y)
        if present.size < 2:
            raise ValueError(
                f"training set contains a single class ({present.tolist()}); "
                "annotate more edges before training"
            )
        weights = _sample_weights(y, len(self.metadata.class_names), self.training_config.class_weighting)
        clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) if self.backend != "xgboost" else X
        if self.backend == "random_forest":
            self.model.fit(clean, y)
        elif self.backend == "hist_gradient_boosting":
            self.model.fit(clean, y, sample_weight=weights)
        else:
            self.model.fit(clean, y, sample_weight=weights)
        self.metadata.training_rows = int(X.shape[0])
        self.metadata.class_counts = {
            self.metadata.class_names[int(index)]: int(count)
            for index, count in zip(*np.unique(y, return_counts=True))
        }
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        clean = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0) if self.backend != "xgboost" else X
        proba = np.asarray(self.model.predict_proba(clean), dtype=np.float64)
        expected = len(self.metadata.class_names)
        if proba.shape[1] != expected:  # a class was absent from the training fold
            full = np.zeros((proba.shape[0], expected), dtype=np.float64)
            known = getattr(self.model, "classes_", np.arange(proba.shape[1]))
            for column, class_index in enumerate(known):
                full[:, int(class_index)] = proba[:, column]
            return full
        return proba

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.predict_proba(X), axis=1)

    def feature_importances(self) -> Dict[str, float]:
        raw = getattr(self.model, "feature_importances_", None)
        if raw is None:
            from sklearn.inspection import permutation_importance  # noqa: F401  (documented alternative)

            logger.warning("%s exposes no feature_importances_", self.backend)
            return {}
        return {name: float(value) for name, value in zip(self.metadata.feature_names, raw)}

    def save(self, path: Path) -> Path:
        """Persist the fitted model next to a JSON metadata sidecar."""
        import joblib

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path)
        path.with_suffix(".json").write_text(
            json.dumps(self.metadata.as_dict(), indent=2), encoding="utf-8"
        )
        logger.info("baseline model saved: %s (%s)", path, self.backend)
        return path

    @classmethod
    def load(cls, path: Path, model_config: ModelConfig, training_config: TrainingConfig) -> "BaselineClassifier":
        """Restore a saved model together with its feature and class names."""
        import joblib

        path = Path(path)
        meta_path = path.with_suffix(".json")
        if not meta_path.is_file():
            raise FileNotFoundError(f"model metadata missing: {meta_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        instance = cls.__new__(cls)
        instance.model_config = model_config
        instance.training_config = training_config
        instance.backend = meta["backend"]
        instance.metadata = BaselineMetadata(
            backend=meta["backend"],
            feature_names=list(meta["feature_names"]),
            class_names=list(meta["class_names"]),
            config_hash=meta.get("config_hash", ""),
            random_seed=int(meta.get("random_seed", model_config.random_seed)),
            training_rows=int(meta.get("training_rows", 0)),
            class_counts=dict(meta.get("class_counts", {})),
        )
        instance.model = joblib.load(path)
        return instance
