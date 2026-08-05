"""Pure-PyTorch graph neural network training for edge classification.

The feature table stores one row per skeleton edge and the terminal graph-node
ids in ``u`` and ``v``.  This module turns each image into a *line graph*: an
edge-feature row is a node, and two rows are connected when their skeleton
edges share a terminal node.  A small GINE-style message-passing network then
uses the local topology in addition to the handcrafted features.

The implementation intentionally uses only PyTorch.  The previous CLI
advertised a ``torch-geometric`` GNN while the module was missing altogether;
keeping the graph construction here makes the command runnable in the same
environment as the original RIFT project and avoids an optional binary
dependency for this relatively small line graph.
"""
from __future__ import annotations

import logging
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ..config import TopologyConfig
from ..logging_utils import environment_report, write_json
from ..visualization.reports import confusion_matrix_figure, write_report
from .dataset import numeric_feature_columns, prepare_training_frame
from .metrics import EvaluationResult, aggregate_folds, evaluate
from .splits import SplitReport, split_frame

logger = logging.getLogger(__name__)


APPEARANCE_PREFIXES = (
    "line_gray",
    "side_gray",
    "contrast",
    "side_asymmetry",
    "patch_gray",
    "sample_count",
    "appearance_available",
    "line_saturation",
    "side_saturation",
)


@dataclass(frozen=True)
class EdgeGraphSample:
    """One image represented as a graph over its skeleton edges."""

    image_id: str
    panel_id: str
    row_indices: np.ndarray
    features: np.ndarray
    labels: np.ndarray
    edge_index: np.ndarray

    @property
    def node_count(self) -> int:
        return int(self.features.shape[0])


@dataclass(frozen=True)
class FeatureScaler:
    """Train-only feature normalization persisted with a GNN checkpoint."""

    mean: np.ndarray
    scale: np.ndarray

    @classmethod
    def fit(cls, arrays: Iterable[np.ndarray]) -> "FeatureScaler":
        chunks = [np.asarray(array, dtype=np.float32) for array in arrays]
        if not chunks:
            raise ValueError("cannot fit a feature scaler on no graph samples")
        values = np.concatenate(chunks, axis=0)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
        scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
        scale[~np.isfinite(scale) | (scale < 1e-6)] = 1.0
        return cls(mean=mean, scale=scale)

    def transform(self, values: np.ndarray) -> np.ndarray:
        array = np.nan_to_num(
            np.asarray(values, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        return ((array - self.mean) / self.scale).astype(np.float32, copy=False)

    def as_dict(self) -> Dict[str, List[float]]:
        return {"mean": self.mean.astype(float).tolist(), "scale": self.scale.astype(float).tolist()}


@dataclass
class GNNTrainingOutcome:
    """Artifacts and cross-validation results produced by :func:`train_gnn`."""

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
            "folds": [result.as_dict() for result in self.fold_results],
            "model_path": str(self.model_path) if self.model_path else None,
            "report_path": str(self.report_path) if self.report_path else None,
            "notes": self.notes,
        }


def set_torch_seed(seed: int) -> None:
    """Seed Python, NumPy and PyTorch for reproducible small-graph training."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_gnn_features(frame: pd.DataFrame, use_appearance: bool = True) -> List[str]:
    """Select numeric columns while optionally excluding appearance features."""

    columns = numeric_feature_columns(frame)
    if use_appearance:
        return columns
    return [
        column
        for column in columns
        if not any(column.startswith(prefix) for prefix in APPEARANCE_PREFIXES)
    ]


def _as_endpoint(value: object, column: str, row_index: object) -> int:
    try:
        numeric = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"row {row_index}: graph endpoint {column!r} is not numeric") from error
    if not np.isfinite(numeric) or numeric != int(numeric):
        raise ValueError(f"row {row_index}: graph endpoint {column!r} is invalid: {value!r}")
    return int(numeric)


def _line_graph_edges(endpoints: Sequence[Tuple[int, int]]) -> np.ndarray:
    """Return directed line-graph edges for pairs sharing a terminal node."""

    incident: Dict[int, List[int]] = {}
    for local_index, (u, v) in enumerate(endpoints):
        for node_id in {u, v}:
            incident.setdefault(node_id, []).append(local_index)

    pairs: set[Tuple[int, int]] = set()
    for local_indices in incident.values():
        for source in local_indices:
            for target in local_indices:
                if source != target:
                    pairs.add((source, target))
    if not pairs:
        return np.empty((2, 0), dtype=np.int64)
    ordered = sorted(pairs)
    return np.asarray(ordered, dtype=np.int64).T


def build_edge_graphs(
    frame: pd.DataFrame,
    feature_columns: Sequence[str],
    require_labels: bool = True,
) -> List[EdgeGraphSample]:
    """Convert an edge feature frame into one graph sample per image.

    An image must belong to one panel.  If it does not, a grouped split could
    place parts of the same image on opposite sides, so the condition is
    rejected explicitly instead of silently creating a leakage-prone graph.
    """

    required = {"image_id", "panel_id", "u", "v"}
    if require_labels:
        required.add("y")
    missing = sorted(required - set(frame.columns))
    if missing:
        raise KeyError(f"GNN feature frame is missing required column(s): {missing}")
    if not feature_columns:
        raise ValueError("no numeric feature columns available for GNN training")

    samples: List[EdgeGraphSample] = []
    for image_id, group in frame.groupby("image_id", sort=True, dropna=False):
        row_indices = np.asarray(group.index.to_numpy(), dtype=np.int64)
        panel_values = group["panel_id"].astype(str).unique().tolist()
        if len(panel_values) != 1:
            raise ValueError(
                f"image {image_id!r} belongs to multiple panels: {panel_values[:5]}"
            )

        endpoints: List[Tuple[int, int]] = []
        for row_index, row in group.iterrows():
            endpoints.append(
                (
                    _as_endpoint(row["u"], "u", row_index),
                    _as_endpoint(row["v"], "v", row_index),
                )
            )

        values = group.loc[:, list(feature_columns)].to_numpy(dtype=np.float32)
        values = np.nan_to_num(values, nan=0.0, posinf=0.0, neginf=0.0)
        labels = (
            group["y"].to_numpy(dtype=np.int64)
            if "y" in group.columns
            else np.zeros(len(group), dtype=np.int64)
        )
        samples.append(
            EdgeGraphSample(
                image_id=str(image_id),
                panel_id=panel_values[0],
                row_indices=row_indices,
                features=values,
                labels=labels,
                edge_index=_line_graph_edges(endpoints),
            )
        )
    return samples


class _GINEBlock(nn.Module):
    """A compact GINE-style block with normalized neighbourhood aggregation."""

    def __init__(self, hidden_dim: int, dropout: float) -> None:
        super().__init__()
        self.epsilon = nn.Parameter(torch.zeros(1))
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden: Tensor, edge_index: Tensor) -> Tensor:
        messages = torch.zeros_like(hidden)
        degree = torch.zeros(hidden.shape[0], dtype=hidden.dtype, device=hidden.device)
        if edge_index.numel():
            source, target = edge_index
            messages.index_add_(0, target, hidden[source])
            degree.index_add_(0, target, torch.ones_like(target, dtype=hidden.dtype))
        messages = messages / degree.clamp_min(1.0).unsqueeze(1)
        updated = self.mlp((1.0 + self.epsilon) * hidden + messages)
        return self.norm(hidden + self.dropout(F.relu(updated)))


class EdgeGINE(nn.Module):
    """Message-passing classifier whose nodes are skeleton edges."""

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        class_count: int,
        num_layers: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if input_dim < 1:
            raise ValueError("input_dim must be positive")
        if hidden_dim < 1 or class_count < 2:
            raise ValueError("hidden_dim must be positive and class_count must be at least 2")
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")
        self.input_dim = int(input_dim)
        self.hidden_dim = int(hidden_dim)
        self.class_count = int(class_count)
        self.num_layers = int(num_layers)
        self.dropout_rate = float(dropout)
        self.input = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.layers = nn.ModuleList(
            _GINEBlock(hidden_dim, dropout) for _ in range(num_layers)
        )
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, class_count),
        )

    def forward(self, features: Tensor, edge_index: Tensor) -> Tensor:
        hidden = self.input(features)
        for layer in self.layers:
            hidden = layer(hidden, edge_index)
        return self.head(hidden)


def _class_weights(labels: np.ndarray, class_count: int, mode: str) -> Optional[Tensor]:
    if mode != "balanced":
        return None
    counts = np.bincount(labels.astype(np.int64), minlength=class_count).astype(np.float32)
    counts[counts <= 0] = 1.0
    weights = float(len(labels)) / (float(class_count) * counts)
    return torch.from_numpy(weights.astype(np.float32))


def _loss(
    logits: Tensor,
    labels: Tensor,
    weights: Optional[Tensor],
    focal_gamma: float,
) -> Tensor:
    per_item = F.cross_entropy(logits, labels, weight=weights, reduction="none")
    if focal_gamma > 0:
        probabilities = torch.softmax(logits.detach(), dim=1)
        true_probability = probabilities.gather(1, labels[:, None]).squeeze(1)
        per_item = per_item * (1.0 - true_probability).clamp_min(1e-6).pow(focal_gamma)
    return per_item.mean()


def _resolve_device(device: Optional[str]) -> torch.device:
    if device:
        requested = torch.device(device)
        if requested.type == "cuda" and not torch.cuda.is_available():
            raise ValueError("requested CUDA for GNN training, but CUDA is unavailable")
        return requested
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _fit_model(
    samples: Sequence[EdgeGraphSample],
    input_dim: int,
    class_count: int,
    config: TopologyConfig,
    epochs: int,
    seed: int,
    device: torch.device,
) -> Tuple[EdgeGINE, FeatureScaler, List[float]]:
    if not samples:
        raise ValueError("cannot train GNN with no image graphs")
    labels = np.concatenate([sample.labels for sample in samples])
    if np.unique(labels).size < 2:
        raise ValueError(
            "GNN training needs at least two classes in the training partition; "
            f"found {np.unique(labels).tolist()}"
        )
    scaler = FeatureScaler.fit(sample.features for sample in samples)
    model = EdgeGINE(
        input_dim=input_dim,
        hidden_dim=config.gnn.hidden_dim,
        class_count=class_count,
        num_layers=config.gnn.num_layers,
        dropout=config.gnn.dropout,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.gnn.learning_rate,
        weight_decay=config.gnn.weight_decay,
    )
    weights = _class_weights(labels, class_count, config.training.class_weighting)
    if weights is not None:
        weights = weights.to(device)
    history: List[float] = []
    order_rng = np.random.default_rng(seed)

    for epoch in range(epochs):
        model.train()
        order = order_rng.permutation(len(samples))
        losses: List[float] = []
        for sample_index in order:
            sample = samples[int(sample_index)]
            features = torch.from_numpy(scaler.transform(sample.features)).to(device)
            labels_tensor = torch.from_numpy(sample.labels).to(device)
            edge_index = torch.from_numpy(sample.edge_index).long().to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(features, edge_index)
            loss = _loss(logits, labels_tensor, weights, config.gnn.focal_gamma)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            losses.append(float(loss.detach().cpu()))
        mean_loss = float(np.mean(losses)) if losses else float("nan")
        history.append(mean_loss)
        logger.debug("GNN epoch %d/%d: loss=%.6f", epoch + 1, epochs, mean_loss)
    return model, scaler, history


def _predict_samples(
    model: EdgeGINE,
    scaler: FeatureScaler,
    samples: Sequence[EdgeGraphSample],
    device: torch.device,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    row_indices: List[np.ndarray] = []
    predictions: List[np.ndarray] = []
    probabilities: List[np.ndarray] = []
    with torch.no_grad():
        for sample in samples:
            features = torch.from_numpy(scaler.transform(sample.features)).to(device)
            edge_index = torch.from_numpy(sample.edge_index).long().to(device)
            logits = model(features, edge_index)
            proba = torch.softmax(logits, dim=1).cpu().numpy()
            row_indices.append(sample.row_indices)
            predictions.append(np.argmax(proba, axis=1).astype(np.int64))
            probabilities.append(proba.astype(np.float64))
    if not row_indices:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty((0, model.class_count), dtype=np.float64),
        )
    rows = np.concatenate(row_indices)
    order = np.argsort(rows)
    return rows[order], np.concatenate(predictions)[order], np.concatenate(probabilities)[order]


def _partition_samples(
    samples: Sequence[EdgeGraphSample],
    row_indices: Sequence[int],
    opposite_indices: Optional[Sequence[int]] = None,
) -> List[EdgeGraphSample]:
    selected = set(int(index) for index in row_indices)
    opposite_values = opposite_indices if opposite_indices is not None else ()
    opposite = set(int(index) for index in opposite_values)
    result: List[EdgeGraphSample] = []
    for sample in samples:
        rows = set(int(index) for index in sample.row_indices)
        if rows & selected and rows & opposite:
            raise ValueError(
                f"image {sample.image_id!r} is split across train and validation rows; "
                "grouping must be consistent at image level"
            )
        if rows and rows <= selected:
            result.append(sample)
    return result


def _evaluate_partition(
    model: EdgeGINE,
    scaler: FeatureScaler,
    samples: Sequence[EdgeGraphSample],
    frame: pd.DataFrame,
    config: TopologyConfig,
    class_names: Sequence[str],
    device: torch.device,
) -> EvaluationResult:
    rows, predictions, probabilities = _predict_samples(model, scaler, samples, device)
    if rows.size == 0:
        return evaluate([], [], class_names)
    y_true = frame.iloc[rows]["y"].to_numpy(dtype=np.int64)
    lengths = (
        frame.iloc[rows]["length_px"].to_numpy(dtype=float)
        if "length_px" in frame.columns
        else None
    )
    confidence = probabilities.max(axis=1)
    return evaluate(
        y_true,
        predictions,
        class_names,
        lengths=lengths,
        abstained=confidence < config.model.minimum_prediction_confidence,
    )


def _checkpoint_payload(
    model: EdgeGINE,
    scaler: FeatureScaler,
    feature_columns: Sequence[str],
    class_names: Sequence[str],
    config: TopologyConfig,
    training_rows: int,
    history: Sequence[float],
) -> Dict[str, object]:
    return {
        "format_version": 1,
        "model_type": "edge_gine_pytorch",
        "state_dict": model.state_dict(),
        "model": {
            "input_dim": model.input_dim,
            "hidden_dim": model.hidden_dim,
            "class_count": model.class_count,
            "num_layers": model.num_layers,
            "dropout": model.dropout_rate,
        },
        "feature_names": list(feature_columns),
        "class_names": list(class_names),
        "feature_scaler": scaler.as_dict(),
        "config_hash": config.config_hash(scope="features"),
        "random_seed": config.model.random_seed,
        "training_rows": int(training_rows),
        "loss_history": [float(value) for value in history],
    }


def load_gnn_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> Dict[str, object]:
    """Load and validate a saved GNN checkpoint for downstream inference."""

    checkpoint = torch.load(Path(path), map_location=map_location)
    if not isinstance(checkpoint, Mapping) or checkpoint.get("model_type") != "edge_gine_pytorch":
        raise ValueError(f"{path}: not a supported edge GNN checkpoint")
    required = {"state_dict", "model", "feature_names", "class_names", "feature_scaler"}
    missing = sorted(required - set(checkpoint))
    if missing:
        raise ValueError(f"{path}: checkpoint missing {missing}")
    return dict(checkpoint)


@dataclass(frozen=True)
class GNNMetadata:
    """Inference metadata exposed through the same shape as baseline metadata."""

    feature_names: List[str]
    class_names: List[str]
    config_hash: str
    random_seed: int
    training_rows: int

    def as_dict(self) -> Dict[str, object]:
        return {
            "backend": "edge_gine_pytorch",
            "feature_names": self.feature_names,
            "class_names": self.class_names,
            "config_hash": self.config_hash,
            "random_seed": self.random_seed,
            "training_rows": self.training_rows,
        }


class GNNClassifier:
    """Load a saved edge-GINE checkpoint and produce per-edge probabilities."""

    def __init__(self, checkpoint: Mapping[str, object], device: Optional[str] = None) -> None:
        model_payload = checkpoint["model"]
        if not isinstance(model_payload, Mapping):
            raise ValueError("GNN checkpoint model metadata is invalid")
        feature_names = [str(name) for name in checkpoint["feature_names"]]  # type: ignore[index]
        class_names = [str(name) for name in checkpoint["class_names"]]  # type: ignore[index]
        self.metadata = GNNMetadata(
            feature_names=feature_names,
            class_names=class_names,
            config_hash=str(checkpoint.get("config_hash", "")),
            random_seed=int(checkpoint.get("random_seed", 0)),
            training_rows=int(checkpoint.get("training_rows", 0)),
        )
        self.device = _resolve_device(device)
        self.model = EdgeGINE(
            input_dim=int(model_payload["input_dim"]),
            hidden_dim=int(model_payload["hidden_dim"]),
            class_count=int(model_payload["class_count"]),
            num_layers=int(model_payload["num_layers"]),
            dropout=float(model_payload.get("dropout", 0.0)),
        ).to(self.device)
        self.model.load_state_dict(checkpoint["state_dict"])  # type: ignore[arg-type]
        self.model.eval()
        scaler_payload = checkpoint["feature_scaler"]
        if not isinstance(scaler_payload, Mapping):
            raise ValueError("GNN checkpoint feature scaler is invalid")
        self.scaler = FeatureScaler(
            mean=np.asarray(scaler_payload["mean"], dtype=np.float32),
            scale=np.asarray(scaler_payload["scale"], dtype=np.float32),
        )
        if len(self.metadata.feature_names) != self.model.input_dim:
            raise ValueError("GNN checkpoint feature names do not match model input_dim")
        if self.scaler.mean.shape != (self.model.input_dim,) or self.scaler.scale.shape != (
            self.model.input_dim,
        ):
            raise ValueError("GNN checkpoint scaler shape does not match model input_dim")

    @classmethod
    def load(cls, path: Path, device: Optional[str] = None) -> "GNNClassifier":
        return cls(load_gnn_checkpoint(path), device=device)

    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        """Predict rows in ``frame`` while preserving their original order."""

        missing = [name for name in self.metadata.feature_names if name not in frame.columns]
        if missing:
            raise KeyError(f"GNN feature table is missing columns: {missing[:10]}")
        working = frame.reset_index(drop=True).copy()
        working["y"] = 0
        samples = build_edge_graphs(
            working,
            self.metadata.feature_names,
            require_labels=False,
        )
        rows, _, probabilities = _predict_samples(self.model, self.scaler, samples, self.device)
        if len(rows) != len(working) or not np.array_equal(rows, np.arange(len(working))):
            raise ValueError("GNN prediction rows could not be aligned with the feature table")
        return probabilities


def train_gnn(
    config: TopologyConfig,
    features: pd.DataFrame,
    output_dir: Optional[Path] = None,
    epochs: Optional[int] = None,
    device: Optional[str] = None,
) -> GNNTrainingOutcome:
    """Grouped-CV train and final-fit of the edge GNN.

    The split is performed on ``config.training.group_column`` exactly like
    the baseline, so rows from the same panel never cross a fold.  The final
    checkpoint contains feature names and normalization statistics, making a
    later inference adapter able to reject incompatible feature tables.
    """

    set_torch_seed(config.model.random_seed)
    output_dir = Path(output_dir) if output_dir else config.output_dir / "gnn"
    output_dir.mkdir(parents=True, exist_ok=True)
    frame, class_names = prepare_training_frame(features, config)
    feature_columns = select_gnn_features(frame, config.gnn.use_appearance_features)
    if not feature_columns:
        raise ValueError("no GNN feature columns remain after appearance filtering")
    group_column = config.training.group_column
    if group_column not in frame.columns:
        raise KeyError(f"group column {group_column!r} not in the feature table")
    if frame["y"].nunique() < 2:
        raise ValueError("GNN training needs at least two labelled classes")

    samples = build_edge_graphs(frame, feature_columns)
    if not samples:
        raise ValueError("no image graphs can be built from the feature table")
    target_epochs = config.gnn.epochs if epochs is None else int(epochs)
    if target_epochs < 1:
        raise ValueError(f"epochs must be at least 1, got {target_epochs}")
    resolved_device = _resolve_device(device)
    logger.info(
        "GNN: %d image graph(s), %d edge nodes, %d features, device=%s, epochs=%d",
        len(samples),
        len(frame),
        len(feature_columns),
        resolved_device,
        target_epochs,
    )

    folds, split_report = split_frame(
        frame,
        label_column="y",
        group_column=group_column,
        n_splits=config.training.n_splits,
        seed=config.model.random_seed,
    )
    notes: List[str] = list(split_report.warnings)
    fold_results: List[EvaluationResult] = []
    for fold_index, (train_index, test_index) in enumerate(folds):
        train_samples = _partition_samples(samples, train_index, test_index)
        test_samples = _partition_samples(samples, test_index, train_index)
        if not train_samples or not test_samples:
            message = f"fold {fold_index}: skipped (empty image partition)"
            notes.append(message)
            logger.warning(message)
            continue
        train_labels = np.concatenate([sample.labels for sample in train_samples])
        if np.unique(train_labels).size < 2:
            message = f"fold {fold_index}: skipped (training partition has one class)"
            notes.append(message)
            logger.warning(message)
            continue
        model, scaler, _ = _fit_model(
            train_samples,
            len(feature_columns),
            len(class_names),
            config,
            target_epochs,
            config.model.random_seed + fold_index,
            resolved_device,
        )
        result = _evaluate_partition(
            model,
            scaler,
            test_samples,
            frame,
            config,
            class_names,
            resolved_device,
        )
        fold_results.append(result)
        logger.info("GNN fold %d: %s", fold_index, result.summary_line())

    if not fold_results:
        raise ValueError("every GNN fold was skipped; not enough grouped labelled data to evaluate")
    aggregate = aggregate_folds(fold_results)

    final_model, final_scaler, history = _fit_model(
        samples,
        len(feature_columns),
        len(class_names),
        config,
        target_epochs,
        config.model.random_seed,
        resolved_device,
    )
    checkpoint_path = output_dir / "gnn_model.pt"
    torch.save(
        _checkpoint_payload(
            final_model,
            final_scaler,
            feature_columns,
            class_names,
            config,
            len(frame),
            history,
        ),
        checkpoint_path,
    )
    write_json(
        output_dir / "gnn_training_history.json",
        {"epochs": target_epochs, "loss": history, "device": str(resolved_device)},
    )
    confusion_matrix_figure(
        np.asarray(aggregate["confusion_sum"], dtype=int),
        class_names,
        output_dir / "confusion_matrix.png",
        title=f"Edge GINE - {split_report.strategy}",
    )
    report_path = write_report(
        output_dir / "gnn_report.md",
        title="Graph neural network edge classifier",
        sections={
            "Data": {
                "labelled_edges": len(frame),
                "image_graphs": len(samples),
                "feature_count": len(feature_columns),
                "class_counts": frame["label"].value_counts().to_dict(),
                "group_column": group_column,
                "group_count": split_report.group_count,
            },
            "Model": {
                "type": "pure PyTorch edge GINE",
                "hidden_dim": config.gnn.hidden_dim,
                "num_layers": config.gnn.num_layers,
                "dropout": config.gnn.dropout,
                "device": str(resolved_device),
                "epochs": target_epochs,
            },
            "Split": split_report.as_dict(),
            "Cross-validated metrics": {
                key: value for key, value in aggregate.items() if key != "confusion_sum"
            },
            "Per-fold": [result.summary_line() for result in fold_results],
            "Notes": notes or ["none"],
            "Environment": environment_report(),
        },
    )
    outcome = GNNTrainingOutcome(
        fold_results=fold_results,
        aggregate=aggregate,
        split_report=split_report,
        class_names=class_names,
        feature_columns=feature_columns,
        model_path=checkpoint_path,
        report_path=report_path,
        notes=notes,
    )
    write_json(output_dir / "gnn_training_outcome.json", outcome.as_dict())
    return outcome


__all__ = [
    "EdgeGINE",
    "EdgeGraphSample",
    "FeatureScaler",
    "GNNClassifier",
    "GNNMetadata",
    "GNNTrainingOutcome",
    "build_edge_graphs",
    "load_gnn_checkpoint",
    "select_gnn_features",
    "set_torch_seed",
    "train_gnn",
]
