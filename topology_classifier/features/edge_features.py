"""Per-edge geometric, width, curvature and probability descriptors."""
from __future__ import annotations

import logging
from typing import Dict, Optional

import numpy as np

from ..config import GraphConfig
from ..graph.graph_types import SkeletonEdge, SkeletonGraph
from .orientation import path_orientation_stats, segment_angles

logger = logging.getLogger(__name__)

NAN = float("nan")


def _resample_indices(count: int, step: int) -> np.ndarray:
    if count <= 2 or step <= 1:
        return np.arange(count)
    indices = np.arange(0, count, step)
    if indices[-1] != count - 1:
        indices = np.append(indices, count - 1)
    return indices


def curvature_stats(path: np.ndarray, step: int) -> Dict[str, float]:
    """Turning-angle curvature along a resampled polyline.

    Returns mean/max absolute turning per unit length and the total signed
    turning, which separates a wavy line from a consistently bending arc.
    """
    if path.shape[0] < 3:
        return {
            "curvature_mean": 0.0,
            "curvature_max": 0.0,
            "curvature_std": 0.0,
            "total_turning_deg": 0.0,
            "net_turning_deg": 0.0,
        }
    points = path[_resample_indices(path.shape[0], step)].astype(np.float64)
    deltas = np.diff(points, axis=0)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    keep = lengths > 1e-9
    deltas, lengths = deltas[keep], lengths[keep]
    if deltas.shape[0] < 2:
        return {
            "curvature_mean": 0.0,
            "curvature_max": 0.0,
            "curvature_std": 0.0,
            "total_turning_deg": 0.0,
            "net_turning_deg": 0.0,
        }
    headings = np.arctan2(deltas[:, 0], deltas[:, 1])
    turning = np.diff(headings)
    turning = (turning + np.pi) % (2 * np.pi) - np.pi
    span = (lengths[:-1] + lengths[1:]) / 2.0
    curvature = np.abs(turning) / np.maximum(span, 1e-9)
    return {
        "curvature_mean": float(curvature.mean()),
        "curvature_max": float(curvature.max()),
        "curvature_std": float(curvature.std()),
        "total_turning_deg": float(np.degrees(np.abs(turning).sum())),
        "net_turning_deg": float(np.degrees(turning.sum())),
    }


def straightness_residual(path: np.ndarray) -> float:
    """RMS deviation from the total-least-squares line fit, in pixels."""
    if path.shape[0] < 3:
        return 0.0
    points = path.astype(np.float64)
    centred = points - points.mean(axis=0)
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    normal = vt[-1]
    return float(np.sqrt(np.mean((centred @ normal) ** 2)))


def _series_stats(values: np.ndarray, prefix: str) -> Dict[str, float]:
    if values.size == 0:
        return {
            f"{prefix}_mean": NAN,
            f"{prefix}_std": NAN,
            f"{prefix}_min": NAN,
            f"{prefix}_max": NAN,
            f"{prefix}_p10": NAN,
            f"{prefix}_p90": NAN,
            f"{prefix}_cv": NAN,
        }
    mean = float(values.mean())
    std = float(values.std())
    return {
        f"{prefix}_mean": mean,
        f"{prefix}_std": std,
        f"{prefix}_min": float(values.min()),
        f"{prefix}_max": float(values.max()),
        f"{prefix}_p10": float(np.percentile(values, 10)),
        f"{prefix}_p90": float(np.percentile(values, 90)),
        f"{prefix}_cv": float(std / mean) if abs(mean) > 1e-9 else NAN,
    }


def edge_geometry_features(edge: SkeletonEdge, config: GraphConfig) -> Dict[str, float]:
    """Length, tortuosity, curvature, orientation and bounding box of one edge."""
    path = edge.path
    length = edge.length_px()
    start = path[0].astype(np.float64)
    end = path[-1].astype(np.float64)
    euclidean = float(np.hypot(*(end - start)))
    rows, cols = path[:, 0], path[:, 1]
    bbox_height = float(rows.max() - rows.min() + 1)
    bbox_width = float(cols.max() - cols.min() + 1)
    bbox_diagonal = float(np.hypot(bbox_height, bbox_width))

    features: Dict[str, float] = {
        "length_px": length,
        "pixel_count": float(path.shape[0]),
        "euclidean_px": euclidean,
        "tortuosity": float(length / euclidean) if euclidean > 1e-6 else NAN,
        "straightness": float(euclidean / length) if length > 1e-6 else NAN,
        "straightness_residual": straightness_residual(path),
        "bbox_height": bbox_height,
        "bbox_width": bbox_width,
        "bbox_diagonal": bbox_diagonal,
        "bbox_fill_ratio": float(length / bbox_diagonal) if bbox_diagonal > 1e-6 else NAN,
        "is_loop": float(edge.is_loop),
    }
    features.update(curvature_stats(path, config.curvature_resample_step))

    stats = path_orientation_stats(path, bins=config.orientation_bins, step=config.curvature_resample_step)
    features.update(stats.as_dict("orientation"))

    angles = segment_angles(path, step=config.curvature_resample_step)
    features["orientation_range_deg"] = (
        float(np.degrees(angles.max() - angles.min())) if angles.size else NAN
    )
    return features


def edge_width_features(
    edge: SkeletonEdge,
    distance: Optional[np.ndarray],
    config: GraphConfig,
) -> Dict[str, float]:
    """Line width along the edge, taken from the Euclidean distance transform."""
    if distance is None:
        return _series_stats(np.empty(0), "width")
    stride = max(1, config.width_sample_stride)
    path = edge.path[::stride]
    widths = 2.0 * distance[path[:, 0], path[:, 1]].astype(np.float64)
    features = _series_stats(widths, "width")
    length = edge.length_px()
    features["width_length_ratio"] = float(features["width_mean"] / length) if length > 1e-6 else NAN
    return features


def edge_probability_features(edge: SkeletonEdge, probability: Optional[np.ndarray]) -> Dict[str, float]:
    """Segmentation confidence sampled along the edge.

    Returns NaNs when Stage-1 only produced a binary mask; the caller records
    ``probability_available = False`` so this is never mistaken for low
    confidence.
    """
    if probability is None:
        return {**_series_stats(np.empty(0), "prob"), "prob_available": 0.0}
    values = probability[edge.path[:, 0], edge.path[:, 1]].astype(np.float64)
    return {**_series_stats(values, "prob"), "prob_available": 1.0}


def edge_topology_features(graph: SkeletonGraph, edge: SkeletonEdge) -> Dict[str, float]:
    """Degrees and border status of the two terminal nodes, order-invariant."""
    node_u = graph.nodes.get(edge.u)
    node_v = graph.nodes.get(edge.v)
    degrees = [float(node_u.degree) if node_u else NAN, float(node_v.degree) if node_v else NAN]
    finite = [d for d in degrees if not np.isnan(d)]
    border = [bool(node_u and node_u.is_border_node), bool(node_v and node_v.is_border_node)]
    endpoint_count = sum(1 for node in (node_u, node_v) if node is not None and node.degree == 1)
    return {
        "degree_min": float(min(finite)) if finite else NAN,
        "degree_max": float(max(finite)) if finite else NAN,
        "degree_sum": float(sum(finite)) if finite else NAN,
        "free_end_count": float(endpoint_count),
        "touches_border": float(any(border)),
        "both_ends_border": float(all(border)),
        "parallel_edge_count": float(max(0, graph.graph.number_of_edges(edge.u, edge.v) - 1)),
    }
