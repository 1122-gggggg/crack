"""Dataset assembly, grouped splits, metrics and training entry points."""
from .dataset import (
    DatasetBuildResult,
    attach_csv_labels,
    build_feature_dataset,
    feature_matrix,
    numeric_feature_columns,
    prepare_training_frame,
)
from .metrics import EvaluationResult, aggregate_folds, evaluate
from .splits import SplitReport, assert_no_group_leakage, grouped_folds, grouped_holdout, split_frame
from .train_baseline import TrainingOutcome, set_seed, train_baseline
from .train_gnn import (
    EdgeGINE,
    EdgeGraphSample,
    FeatureScaler,
    GNNClassifier,
    GNNMetadata,
    GNNTrainingOutcome,
    build_edge_graphs,
    load_gnn_checkpoint,
    select_gnn_features,
    train_gnn,
)

__all__ = [
    "DatasetBuildResult",
    "EdgeGINE",
    "EdgeGraphSample",
    "EvaluationResult",
    "FeatureScaler",
    "GNNClassifier",
    "GNNMetadata",
    "GNNTrainingOutcome",
    "SplitReport",
    "TrainingOutcome",
    "aggregate_folds",
    "assert_no_group_leakage",
    "attach_csv_labels",
    "build_feature_dataset",
    "build_edge_graphs",
    "evaluate",
    "feature_matrix",
    "grouped_folds",
    "grouped_holdout",
    "load_gnn_checkpoint",
    "numeric_feature_columns",
    "prepare_training_frame",
    "select_gnn_features",
    "set_seed",
    "split_frame",
    "train_baseline",
    "train_gnn",
]
