"""Command line interface for the topology classifier.

Usage::

    python -m topology_classifier <command> --config configs/topology.yaml

Commands: ``build-graphs``, ``export-edge-review``, ``build-features``,
``train-baseline``, ``train-gnn``, ``infer``, ``evaluate``, ``visualize-graph``.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import yaml

from .config import TopologyConfig, dump_config, load_config
from .logging_utils import ErrorJournal, environment_report, setup_logging, write_json

logger = logging.getLogger("topology_classifier")

DEFAULT_CONFIG = Path("configs/topology.yaml")
FEATURE_FILENAME = "edge_features.parquet"


def _parse_overrides(items: Sequence[str]) -> Dict[str, Any]:
    """Turn ``section.key=value`` strings into a nested override mapping."""
    overrides: Dict[str, Any] = {}
    for item in items:
        if "=" not in item or "." not in item.split("=", 1)[0]:
            raise ValueError(f"invalid --set {item!r}; expected section.key=value")
        target, raw = item.split("=", 1)
        section, key = target.split(".", 1)
        overrides.setdefault(section, {})[key] = yaml.safe_load(raw)
    return overrides


def _load(args: argparse.Namespace) -> TopologyConfig:
    config = load_config(args.config, overrides=_parse_overrides(args.set or []))
    setup_logging(args.log_level or config.runtime.log_level)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(
        "config=%s hash=%s (graph=%s features=%s)",
        config.source_path or "<defaults>",
        config.config_hash(),
        config.config_hash(scope="graph"),
        config.config_hash(scope="features"),
    )
    return config


def _journal(config: TopologyConfig) -> ErrorJournal:
    return ErrorJournal(config.output_dir / config.runtime.errors_filename)


def _features_path(config: TopologyConfig, explicit: Optional[str]) -> Path:
    if explicit:
        return Path(explicit)
    return config.output_dir / "features" / FEATURE_FILENAME


def _read_features(path: Path):
    import pandas as pd

    candidates = [path]
    if path.suffix == ".parquet":
        candidates.append(path.with_suffix(".csv"))
    for candidate in candidates:
        if not candidate.is_file():
            continue
        if candidate.suffix == ".parquet":
            return pd.read_parquet(candidate)
        return pd.read_csv(candidate, dtype={"image_id": str, "panel_id": str})
    raise FileNotFoundError(
        f"feature table not found at {path}; run `build-features` first"
    )


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def command_build_graphs(args: argparse.Namespace) -> int:
    from .io.dataset_adapter import DatasetAdapter
    from .pipeline import build_all_graphs, cache_directory

    config = _load(args)
    journal = _journal(config)
    adapter = DatasetAdapter(config)
    artifacts = build_all_graphs(
        config,
        adapter=adapter,
        journal=journal,
        use_cache=not args.no_cache,
        limit=args.limit,
    )
    if not artifacts:
        logger.error("no graphs were built; see %s", journal.path)
        return 1

    rows = [
        {"image_id": a.image_id, "panel_id": a.record.panel_id, "from_cache": a.from_cache, **a.stats}
        for a in artifacts
    ]
    summary_dir = config.output_dir / "graphs"
    write_json(summary_dir / "graph_summary.json", rows)
    try:
        import pandas as pd

        pd.DataFrame(rows).to_csv(summary_dir / "graph_summary.csv", index=False)
    except (ImportError, ValueError) as error:
        logger.warning("could not write graph_summary.csv (%s)", error)

    dump_config(config, summary_dir / "resolved_config.yaml")
    write_json(summary_dir / "environment.json", environment_report())
    logger.info(
        "built %d graph(s), %d failure(s), cache=%s",
        len(artifacts),
        journal.count,
        cache_directory(config),
    )
    return 0


def command_export_edge_review(args: argparse.Namespace) -> int:
    from .io.dataset_adapter import DatasetAdapter
    from .pipeline import build_image_graph
    from .visualization.edge_review import export_edge_review, merge_review_csvs, summarize_review

    config = _load(args)
    journal = _journal(config)
    adapter = DatasetAdapter(config)
    out_dir = Path(args.out) if args.out else config.output_dir / "review"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = adapter.records()
    if args.image_id:
        wanted = set(args.image_id)
        records = [r for r in records if r.image_id in wanted]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        logger.error("no images matched; nothing to export")
        return 1

    written: List[Path] = []
    for index, record in enumerate(records, start=1):
        logger.info("[%d/%d] review export for %s", index, len(records), record.image_id)
        try:
            artifacts = build_image_graph(config, record, adapter)
            image = adapter.load_image(record)
            export = export_edge_review(
                artifacts.graph,
                out_dir=out_dir,
                image=image,
                mask=artifacts.mask,
                minimum_length=args.min_length,
                maximum_count=args.max_edges,
                write_crops=not args.no_crops,
                write_overlay=not args.no_overlay,
                seed=config.model.random_seed,
            )
            written.append(export.csv_path)
            logger.info("%s: %d edge(s) -> %s", record.image_id, export.edge_count, export.csv_path)
        except (OSError, ValueError, KeyError, MemoryError) as error:
            journal.record(record.image_id, stage="export_edge_review", error=error)

    if not written:
        logger.error("every image failed; see %s", journal.path)
        return 1
    merged = merge_review_csvs(written, out_dir / "edge_review_all.csv")
    logger.info("review status: %s", summarize_review(merged))
    logger.info("fill in the empty `label` column, then set data.edge_annotations_csv=%s", merged)
    return 0


def command_build_features(args: argparse.Namespace) -> int:
    from .training.dataset import attach_csv_labels, build_feature_dataset

    config = _load(args)
    journal = _journal(config)
    result = build_feature_dataset(
        config, journal=journal, limit=args.limit, use_cache=not args.no_cache
    )
    if result.features.empty:
        logger.error("feature table is empty; see %s", journal.path)
        return 1

    features = attach_csv_labels(result.features, config)
    result.features = features
    target = _features_path(config, args.out)
    written = result.save(target)

    summary = {
        "rows": int(len(features)),
        "images": len(result.per_image),
        "failed_images": result.failed_image_ids,
        "feature_count": len(result.feature_columns),
        "config_hash_features": config.config_hash(scope="features"),
        "label_counts": (
            features["label"].value_counts().to_dict() if "label" in features.columns else {}
        ),
        "path": str(written),
    }
    write_json(target.parent / "feature_summary.json", summary)
    write_json(target.parent / "per_image.json", result.per_image)
    write_json(target.parent / "environment.json", environment_report())
    logger.info("features written: %s (%d rows)", written, len(features))
    logger.info("label distribution: %s", summary["label_counts"])
    return 0


def command_train_baseline(args: argparse.Namespace) -> int:
    from .training.train_baseline import train_baseline

    config = _load(args)
    features = _read_features(_features_path(config, args.features))
    out_dir = Path(args.out) if args.out else config.output_dir / "baseline"
    try:
        outcome = train_baseline(config, features, output_dir=out_dir)
    except (ValueError, KeyError) as error:
        logger.error("training aborted: %s", error)
        return 1
    logger.info("cross-validated: %s", outcome.aggregate)
    logger.info("model: %s | report: %s", outcome.model_path, outcome.report_path)
    return 0


def command_train_gnn(args: argparse.Namespace) -> int:
    config = _load(args)
    try:
        from .training.train_gnn import train_gnn
    except ImportError as error:
        logger.error(
            "GNN training needs PyTorch (%s); "
            "the baseline path (`train-baseline`) runs without it",
            error,
        )
        return 2

    features = _read_features(_features_path(config, args.features))
    out_dir = Path(args.out) if args.out else config.output_dir / "gnn"
    try:
        outcome = train_gnn(
            config,
            features,
            output_dir=out_dir,
            epochs=args.epochs,
            device=args.device,
        )
    except (ValueError, KeyError, RuntimeError, ImportError, OSError) as error:
        logger.error("GNN training aborted: %s", error)
        return 1
    logger.info("cross-validated: %s", outcome.aggregate)
    logger.info("model: %s | report: %s", outcome.model_path, outcome.report_path)
    return 0


def command_infer(args: argparse.Namespace) -> int:
    from .inference.classify_edges import infer_dataset

    config = _load(args)
    journal = _journal(config)
    default_model_dir = "gnn" if args.model_type == "gnn" else "baseline"
    default_model_name = "gnn_model.pt" if args.model_type == "gnn" else "baseline_model.joblib"
    model_path = Path(args.model) if args.model else config.output_dir / default_model_dir / default_model_name
    if not model_path.is_file():
        command = "train-gnn" if args.model_type == "gnn" else "train-baseline"
        logger.error("model not found: %s (run `%s` first)", model_path, command)
        return 1

    predictions = infer_dataset(
        config,
        model_path=model_path,
        journal=journal,
        output_dir=Path(args.out) if args.out else None,
        limit=args.limit,
        write_masks=not args.no_masks,
        model_type=args.model_type,
    )
    if not predictions:
        logger.error("no image produced predictions; see %s", journal.path)
        return 1

    summary = [
        {
            "image_id": p.image_id,
            "edges": int(len(p.frame)),
            "counts": (
                p.frame["predicted_label"].value_counts().to_dict() if not p.frame.empty else {}
            ),
            "pixel_counts": p.raster.pixel_counts if p.raster else {},
        }
        for p in predictions
    ]
    out_dir = Path(args.out) if args.out else config.output_dir / "predictions"
    write_json(out_dir / "inference_summary.json", summary)
    logger.info("predictions written for %d image(s) -> %s", len(predictions), out_dir)
    return 0


def command_evaluate(args: argparse.Namespace) -> int:
    import numpy as np

    from .models.baseline import BaselineClassifier
    from .training.train_gnn import GNNClassifier
    from .training.dataset import feature_matrix, prepare_training_frame
    from .training.metrics import evaluate as evaluate_metrics
    from .visualization.reports import confusion_matrix_figure, write_report

    config = _load(args)
    default_model_dir = "gnn" if args.model_type == "gnn" else "baseline"
    default_model_name = "gnn_model.pt" if args.model_type == "gnn" else "baseline_model.joblib"
    model_path = Path(args.model) if args.model else config.output_dir / default_model_dir / default_model_name
    if not model_path.is_file():
        logger.error("model not found: %s", model_path)
        return 1

    features = _read_features(_features_path(config, args.features))
    try:
        frame, class_names = prepare_training_frame(features, config)
    except ValueError as error:
        logger.error("cannot evaluate: %s", error)
        return 1

    resolved_model_type = args.model_type
    if resolved_model_type == "auto":
        resolved_model_type = "gnn" if model_path.suffix.lower() in {".pt", ".pth"} else "baseline"
    if resolved_model_type == "gnn":
        model = GNNClassifier.load(model_path)
    else:
        model = BaselineClassifier.load(model_path, config.model, config.training)
    if class_names != model.metadata.class_names:
        logger.warning(
            "class list differs (config=%s, model=%s); using the model's order",
            class_names,
            model.metadata.class_names,
        )
        class_names = model.metadata.class_names

    if isinstance(model, GNNClassifier):
        probabilities = model.predict_proba(frame)
    else:
        X = feature_matrix(frame, model.metadata.feature_names)
        probabilities = model.predict_proba(X)
    predictions = np.argmax(probabilities, axis=1)
    abstained = probabilities.max(axis=1) < config.model.minimum_prediction_confidence
    lengths = frame["length_px"].to_numpy(dtype=float) if "length_px" in frame.columns else None
    result = evaluate_metrics(
        frame["y"].to_numpy(dtype=int), predictions, class_names, lengths=lengths, abstained=abstained
    )
    result.notes.append(
        "scores are in-sample unless this feature table was held out from training "
        f"(the model was fitted on {model.metadata.training_rows} rows)"
    )
    logger.info("evaluation: %s", result.summary_line())

    out_dir = Path(args.out) if args.out else config.output_dir / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    confusion_matrix_figure(result.confusion, class_names, out_dir / "confusion_matrix.png")
    frame_out = frame[["image_id", "panel_id", "edge_id", "label"]].copy()
    frame_out["predicted"] = [class_names[int(i)] for i in predictions]
    frame_out["confidence"] = probabilities.max(axis=1)
    frame_out["abstained"] = abstained
    frame_out.to_csv(out_dir / "edge_evaluation.csv", index=False)
    write_report(
        out_dir / "evaluation_report.md",
        title="Edge classifier evaluation",
        sections={
            "Model": model.metadata.as_dict() | {"path": str(model_path)},
            "Metrics": {k: v for k, v in result.as_dict().items() if k != "confusion"},
            "Environment": environment_report(),
        },
    )
    logger.info("evaluation written -> %s", out_dir)
    return 0


def command_visualize_graph(args: argparse.Namespace) -> int:
    import cv2

    from .io.dataset_adapter import DatasetAdapter
    from .pipeline import build_image_graph
    from .visualization.graph_overlay import base_canvas, class_overlay, draw_edge_ids, draw_edges, draw_nodes

    config = _load(args)
    journal = _journal(config)
    adapter = DatasetAdapter(config)
    out_dir = Path(args.out) if args.out else config.output_dir / "visualization"
    out_dir.mkdir(parents=True, exist_ok=True)

    records = adapter.records()
    if args.image_id:
        wanted = set(args.image_id)
        records = [r for r in records if r.image_id in wanted]
    if args.limit is not None:
        records = records[: args.limit]
    if not records:
        logger.error("no images matched; nothing to visualize")
        return 1

    predictions_dir = Path(args.predictions) if args.predictions else config.output_dir / "predictions"
    written = 0
    for record in records:
        try:
            artifacts = build_image_graph(config, record, adapter)
            image = adapter.load_image(record)
            canvas = base_canvas(artifacts.graph.image_shape, image)
            canvas = draw_edges(canvas, artifacts.graph, thickness=args.thickness)
            canvas = draw_nodes(canvas, artifacts.graph)
            if args.edge_ids:
                canvas = draw_edge_ids(canvas, artifacts.graph)
            target = out_dir / f"{record.image_id}_graph.png"
            cv2.imwrite(str(target), canvas)
            written += 1

            csv_path = predictions_dir / f"{record.image_id}_edge_predictions.csv"
            if csv_path.is_file():
                import pandas as pd

                frame = pd.read_csv(csv_path)
                labels = dict(zip(frame["edge_id"].astype(int), frame["predicted_label"]))
                overlay = class_overlay(artifacts.graph.image_shape, artifacts.graph, labels, image)
                cv2.imwrite(str(out_dir / f"{record.image_id}_classes.png"), overlay)

            logger.info("%s: graph overlay -> %s", record.image_id, target)
        except (OSError, ValueError, KeyError, MemoryError) as error:
            journal.record(record.image_id, stage="visualize_graph", error=error)

    return 0 if written else 1


COMMANDS = {
    "build-graphs": command_build_graphs,
    "export-edge-review": command_export_edge_review,
    "build-features": command_build_features,
    "train-baseline": command_train_baseline,
    "train-gnn": command_train_gnn,
    "infer": command_infer,
    "evaluate": command_evaluate,
    "visualize-graph": command_visualize_graph,
}


def _add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="YAML config path")
    parser.add_argument("--log-level", default=None, help="override runtime.log_level")
    parser.add_argument(
        "--set",
        action="append",
        metavar="SECTION.KEY=VALUE",
        help="override a single config value (repeatable)",
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m topology_classifier",
        description="Crack vs craquelure topology classification on top of RIFT line masks",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    graphs = subparsers.add_parser("build-graphs", help="preprocess masks and build skeleton graphs")
    _add_common(graphs)
    graphs.add_argument("--limit", type=int, default=None)
    graphs.add_argument("--no-cache", action="store_true")

    review = subparsers.add_parser("export-edge-review", help="export edges for human labelling")
    _add_common(review)
    review.add_argument("--limit", type=int, default=None)
    review.add_argument("--image-id", action="append", default=None)
    review.add_argument("--min-length", type=float, default=0.0)
    review.add_argument("--max-edges", type=int, default=None)
    review.add_argument("--no-crops", action="store_true")
    review.add_argument("--no-overlay", action="store_true")
    review.add_argument("--out", default=None)

    features = subparsers.add_parser("build-features", help="extract the edge feature table")
    _add_common(features)
    features.add_argument("--limit", type=int, default=None)
    features.add_argument("--no-cache", action="store_true")
    features.add_argument("--out", default=None)

    baseline = subparsers.add_parser("train-baseline", help="grouped CV + final fit of the tree baseline")
    _add_common(baseline)
    baseline.add_argument("--features", default=None)
    baseline.add_argument("--out", default=None)

    gnn = subparsers.add_parser("train-gnn", help="train the PyTorch GINE edge classifier")
    _add_common(gnn)
    gnn.add_argument("--features", default=None)
    gnn.add_argument("--out", default=None)
    gnn.add_argument("--epochs", type=int, default=None)
    gnn.add_argument("--device", default=None, help="PyTorch device, e.g. cpu or cuda")

    infer = subparsers.add_parser("infer", help="classify edges and rasterize a class mask")
    _add_common(infer)
    infer.add_argument("--model", default=None)
    infer.add_argument(
        "--model-type",
        choices=("auto", "baseline", "gnn"),
        default="auto",
        help="model backend; auto uses .joblib for baseline and .pt/.pth for GNN",
    )
    infer.add_argument("--limit", type=int, default=None)
    infer.add_argument("--out", default=None)
    infer.add_argument("--no-masks", action="store_true")

    evaluation = subparsers.add_parser("evaluate", help="score a saved model on a labelled feature table")
    _add_common(evaluation)
    evaluation.add_argument("--model", default=None)
    evaluation.add_argument(
        "--model-type",
        choices=("auto", "baseline", "gnn"),
        default="auto",
        help="model backend; auto uses .joblib for baseline and .pt/.pth for GNN",
    )
    evaluation.add_argument("--features", default=None)
    evaluation.add_argument("--out", default=None)

    visualize = subparsers.add_parser("visualize-graph", help="render skeleton graph overlays")
    _add_common(visualize)
    visualize.add_argument("--limit", type=int, default=None)
    visualize.add_argument("--image-id", action="append", default=None)
    visualize.add_argument("--out", default=None)
    visualize.add_argument("--predictions", default=None, help="directory holding *_edge_predictions.csv")
    visualize.add_argument("--thickness", type=int, default=2)
    visualize.add_argument("--edge-ids", action="store_true")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handler = COMMANDS[args.command]
    try:
        return handler(args)
    except FileNotFoundError as error:
        setup_logging("INFO")
        logger.error("%s", error)
        return 1
    except KeyboardInterrupt:
        logger.warning("interrupted")
        return 130


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
