"""Human-readable summaries of graphs, evaluations and predictions."""
from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence

import numpy as np

logger = logging.getLogger(__name__)


def write_markdown_table(rows: Sequence[Mapping[str, object]], columns: Sequence[str]) -> str:
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join("---" for _ in columns) + " |"
    body = [
        "| " + " | ".join(_format(row.get(column, "")) for column in columns) + " |"
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _format(value: object) -> str:
    if isinstance(value, float):
        return "n/a" if np.isnan(value) else f"{value:.4f}"
    return str(value)


def confusion_matrix_figure(
    matrix: np.ndarray,
    class_names: Sequence[str],
    path: Path,
    title: str = "Confusion matrix",
) -> Optional[Path]:
    """Save a confusion-matrix PNG; returns ``None`` when matplotlib is absent."""
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping %s", path.name)
        return None

    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(1.6 * len(class_names) + 2, 1.6 * len(class_names) + 1.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks(range(len(class_names)), class_names, rotation=45, ha="right")
    axis.set_yticks(range(len(class_names)), class_names)
    axis.set_xlabel("predicted")
    axis.set_ylabel("true")
    axis.set_title(title)
    maximum = matrix.max() if matrix.size else 1
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            axis.text(
                j,
                i,
                int(matrix[i, j]),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > maximum / 2 else "black",
            )
    figure.colorbar(image, ax=axis, shrink=0.8)
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def feature_importance_figure(
    importances: Mapping[str, float],
    path: Path,
    top_k: int = 25,
) -> Optional[Path]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed; skipping %s", path.name)
        return None

    ranked = sorted(importances.items(), key=lambda item: -abs(item[1]))[:top_k]
    if not ranked:
        return None
    names = [name for name, _ in ranked][::-1]
    values = [value for _, value in ranked][::-1]
    path.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(8, 0.32 * len(names) + 1.5))
    axis.barh(names, values, color="#3b6ea5")
    axis.set_xlabel("importance")
    axis.set_title(f"Top {len(names)} features")
    figure.tight_layout()
    figure.savefig(path, dpi=150)
    plt.close(figure)
    return path


def write_report(path: Path, title: str, sections: Mapping[str, object]) -> Path:
    """Write a Markdown report alongside a machine-readable JSON twin."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", ""]
    for heading, content in sections.items():
        lines.append(f"## {heading}")
        lines.append("")
        if isinstance(content, str):
            lines.append(content)
        elif isinstance(content, Mapping):
            for key, value in content.items():
                lines.append(f"- **{key}**: {_format(value)}")
        elif isinstance(content, Sequence):
            for item in content:
                lines.append(f"- {_format(item)}")
        else:
            lines.append(_format(content))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")

    json_path = path.with_suffix(".json")
    json_path.write_text(json.dumps(_jsonable(sections), indent=2, default=str), encoding="utf-8")
    logger.info("report written: %s", path)
    return path


def _jsonable(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def summarize_predictions(labels: Sequence[str], lengths: Sequence[float]) -> Dict[str, object]:
    """Class counts and total skeleton length per predicted class."""
    counts: Dict[str, int] = {}
    length_by_class: Dict[str, float] = {}
    for label, length in zip(labels, lengths):
        counts[label] = counts.get(label, 0) + 1
        length_by_class[label] = length_by_class.get(label, 0.0) + float(length)
    total_length = sum(length_by_class.values())
    return {
        "edge_counts": dict(sorted(counts.items())),
        "length_px": {k: round(v, 1) for k, v in sorted(length_by_class.items())},
        "length_share": {
            k: round(v / total_length, 4) for k, v in sorted(length_by_class.items())
        }
        if total_length > 0
        else {},
    }
