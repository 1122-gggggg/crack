"""Export material for human edge labelling.

Produces, per image:

* ``<image_id>_edge_overlay.png`` -- every edge in a distinct colour with its id
* ``crops/<image_id>_edge_<id>.png`` -- a zoomed context crop per reviewed edge
* ``<image_id>_edge_review.csv`` -- one row per edge with an empty ``label``
  column for the reviewer to fill in

Nothing is pre-filled: the label column ships empty so no synthetic annotation
can leak into training.
"""
from __future__ import annotations

import csv
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from ..graph.graph_types import SkeletonEdge, SkeletonGraph
from ..labels.csv_edge_labels import REVIEW_COLUMNS
from .graph_overlay import base_canvas, draw_edge_ids, draw_edges, draw_nodes

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ReviewExport:
    csv_path: Path
    overlay_path: Optional[Path]
    crop_paths: List[Path]
    edge_count: int


def select_edges_for_review(
    graph: SkeletonGraph,
    minimum_length: float,
    maximum_count: Optional[int],
    seed: int,
) -> List[SkeletonEdge]:
    """Longest-first selection, then a deterministic random sample if capped."""
    candidates = [e for e in graph.edges.values() if e.length_px() >= minimum_length]
    candidates.sort(key=lambda e: (-e.length_px(), e.edge_id))
    if maximum_count is not None and len(candidates) > maximum_count:
        rng = np.random.default_rng(seed)
        keep_head = maximum_count // 2
        head = candidates[:keep_head]
        tail_pool = candidates[keep_head:]
        picked = rng.choice(len(tail_pool), size=maximum_count - keep_head, replace=False)
        candidates = head + [tail_pool[int(i)] for i in sorted(picked)]
        candidates.sort(key=lambda e: e.edge_id)
    return candidates


def _crop_bounds(edge: SkeletonEdge, shape: tuple[int, int], padding: int) -> tuple[int, int, int, int]:
    rows, cols = edge.path[:, 0], edge.path[:, 1]
    row0 = max(0, int(rows.min()) - padding)
    row1 = min(shape[0], int(rows.max()) + padding + 1)
    col0 = max(0, int(cols.min()) - padding)
    col1 = min(shape[1], int(cols.max()) + padding + 1)
    return row0, row1, col0, col1


def write_edge_crops(
    graph: SkeletonGraph,
    edges: Sequence[SkeletonEdge],
    out_dir: Path,
    image: Optional[np.ndarray],
    mask: Optional[np.ndarray],
    padding: int = 48,
    maximum_side: int = 768,
) -> List[Path]:
    """One crop per edge: context in grey, the edge itself highlighted."""
    import cv2

    out_dir.mkdir(parents=True, exist_ok=True)
    background = image if image is not None else (
        (mask.astype(np.uint8) * 255) if mask is not None else None
    )
    if background is None:
        logger.warning("%s: no image or mask available; skipping edge crops", graph.image_id)
        return []

    paths: List[Path] = []
    for edge in edges:
        row0, row1, col0, col1 = _crop_bounds(edge, graph.image_shape, padding)
        patch = background[row0:row1, col0:col1]
        canvas = patch if patch.ndim == 3 else np.repeat(patch[..., None], 3, axis=2)
        canvas = canvas.copy()
        rows = edge.path[:, 0] - row0
        cols = edge.path[:, 1] - col0
        inside = (rows >= 0) & (rows < canvas.shape[0]) & (cols >= 0) & (cols < canvas.shape[1])
        canvas[rows[inside], cols[inside]] = (60, 60, 235)
        scale = maximum_side / max(canvas.shape[:2])
        if scale < 1.0:
            canvas = cv2.resize(canvas, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        target = out_dir / f"{graph.image_id}_edge_{edge.edge_id:06d}.png"
        cv2.imwrite(str(target), canvas)
        paths.append(target)
    return paths


def export_edge_review(
    graph: SkeletonGraph,
    out_dir: Path,
    image: Optional[np.ndarray] = None,
    mask: Optional[np.ndarray] = None,
    minimum_length: float = 0.0,
    maximum_count: Optional[int] = None,
    write_crops: bool = True,
    write_overlay: bool = True,
    seed: int = 42,
) -> ReviewExport:
    """Write the review CSV plus optional overlay and per-edge crops."""
    import cv2

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    edges = select_edges_for_review(graph, minimum_length, maximum_count, seed)
    logger.info(
        "%s: %d/%d edges selected for review (min_length=%.1f)",
        graph.image_id,
        len(edges),
        len(graph.edges),
        minimum_length,
    )

    csv_path = out_dir / f"{graph.image_id}_edge_review.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([*REVIEW_COLUMNS, "length_px", "row", "col"])
        for edge in edges:
            row, col = edge.path[edge.path.shape[0] // 2]
            writer.writerow(
                [
                    graph.image_id,
                    graph.panel_id,
                    edge.component_id,
                    edge.edge_id,
                    "",  # label: to be filled in by the reviewer
                    "",  # confidence
                    "",  # notes
                    round(edge.length_px(), 2),
                    int(row),
                    int(col),
                ]
            )

    overlay_path: Optional[Path] = None
    if write_overlay:
        canvas = base_canvas(graph.image_shape, image if image is not None else _mask_canvas(mask))
        canvas = draw_edges(canvas, graph, thickness=2)
        canvas = draw_nodes(canvas, graph)
        canvas = draw_edge_ids(canvas, graph, edge_ids=[e.edge_id for e in edges])
        overlay_path = out_dir / f"{graph.image_id}_edge_overlay.png"
        cv2.imwrite(str(overlay_path), canvas)

    crop_paths: List[Path] = []
    if write_crops:
        crop_paths = write_edge_crops(graph, edges, out_dir / "crops", image, mask)

    return ReviewExport(
        csv_path=csv_path,
        overlay_path=overlay_path,
        crop_paths=crop_paths,
        edge_count=len(edges),
    )


def _mask_canvas(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    return np.repeat((mask.astype(np.uint8) * 255)[..., None], 3, axis=2)


def merge_review_csvs(paths: Sequence[Path], target: Path) -> Path:
    """Concatenate per-image review CSVs into one annotation workbook."""
    import pandas as pd

    frames = [pd.read_csv(path, dtype={"image_id": str, "panel_id": str}) for path in paths]
    if not frames:
        raise ValueError("no review CSVs to merge")
    merged = pd.concat(frames, ignore_index=True)
    target.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(target, index=False, encoding="utf-8-sig")
    logger.info("merged %d review CSVs -> %s (%d rows)", len(paths), target, len(merged))
    return target


def summarize_review(csv_path: Path) -> Dict[str, int]:
    """Count how many rows are still unlabelled, for progress reporting."""
    import pandas as pd

    frame = pd.read_csv(csv_path, dtype={"image_id": str})
    labels = frame.get("label")
    if labels is None:
        return {"rows": len(frame), "labelled": 0, "unlabelled": len(frame)}
    filled = labels.notna() & (labels.astype(str).str.strip() != "")
    return {"rows": len(frame), "labelled": int(filled.sum()), "unlabelled": int((~filled).sum())}
