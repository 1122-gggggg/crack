"""Node-level descriptors and their order-invariant aggregation onto edges.

Junction geometry is one of the strongest cues available: craquelure tends to
produce near-orthogonal T junctions with three branches, whereas a structural
crack branches at shallow angles or not at all.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

from ..graph.graph_types import (
    NODE_ENDPOINT,
    NODE_ISOLATED,
    NODE_JUNCTION,
    NODE_VIRTUAL_LOOP,
    SkeletonEdge,
    SkeletonGraph,
)
from .orientation import angular_difference_deg

NAN = float("nan")
NODE_TYPES = (NODE_ENDPOINT, NODE_JUNCTION, NODE_VIRTUAL_LOOP, NODE_ISOLATED)


def incident_branch_angle(edge: SkeletonEdge, node_id: int, window: int) -> Optional[float]:
    """Orientation in degrees of the edge as it leaves ``node_id``."""
    path = edge.path
    if path.shape[0] < 2:
        return None
    if edge.u == node_id:
        segment = path[: min(window, path.shape[0])]
        vector = segment[-1].astype(np.float64) - segment[0].astype(np.float64)
    elif edge.v == node_id:
        segment = path[-min(window, path.shape[0]) :]
        vector = segment[0].astype(np.float64) - segment[-1].astype(np.float64)
    else:
        return None
    if float(np.hypot(*vector)) < 1e-9:
        return None
    return float(np.degrees(np.arctan2(vector[0], vector[1])) % 180.0)


def node_features(graph: SkeletonGraph, node_id: int, window: int = 7) -> Dict[str, float]:
    """Degree, type and branch-angle regularity of one node."""
    node = graph.nodes.get(node_id)
    if node is None:
        return {"degree": NAN, "is_border": NAN, "cluster_size": NAN, **_angle_placeholder()}

    incident = graph.incident_edges(node_id)
    angles: List[float] = []
    for edge in incident:
        angle = incident_branch_angle(edge, node_id, window)
        if angle is not None:
            angles.append(angle)
        if edge.is_loop and edge.u == edge.v == node_id:
            other = incident_branch_angle(
                SkeletonEdge(
                    edge_id=edge.edge_id,
                    u=edge.v,
                    v=edge.u,
                    path=edge.path[::-1].copy(),
                    component_id=edge.component_id,
                    is_loop=edge.is_loop,
                ),
                node_id,
                window,
            )
            if other is not None:
                angles.append(other)

    features: Dict[str, float] = {
        "degree": float(node.degree),
        "is_border": float(node.is_border_node),
        "cluster_size": float(len(node.pixels)),
    }
    for node_type in NODE_TYPES:
        features[f"type_{node_type}"] = float(node.node_type == node_type)
    features.update(_pairwise_angle_stats(angles))
    return features


def _angle_placeholder() -> Dict[str, float]:
    return {
        "branch_angle_min": NAN,
        "branch_angle_max": NAN,
        "branch_angle_mean": NAN,
        "branch_angle_std": NAN,
        "branch_angle_orthogonality": NAN,
    }


def _pairwise_angle_stats(angles: List[float]) -> Dict[str, float]:
    if len(angles) < 2:
        return _angle_placeholder()
    differences = [
        angular_difference_deg(angles[i], angles[j])
        for i in range(len(angles))
        for j in range(i + 1, len(angles))
    ]
    array = np.asarray(differences, dtype=np.float64)
    return {
        "branch_angle_min": float(array.min()),
        "branch_angle_max": float(array.max()),
        "branch_angle_mean": float(array.mean()),
        "branch_angle_std": float(array.std()),
        # 1.0 when every branch pair meets at 90 deg, 0.0 when parallel/collinear.
        "branch_angle_orthogonality": float(np.mean(1.0 - np.abs(array - 90.0) / 90.0)),
    }


def aggregate_endpoint_features(graph: SkeletonGraph, edge: SkeletonEdge, window: int = 7) -> Dict[str, float]:
    """Combine the two terminal nodes symmetrically so edge direction cannot leak."""
    left = node_features(graph, edge.u, window)
    right = node_features(graph, edge.v, window)
    combined: Dict[str, float] = {}
    for key in sorted(set(left) | set(right)):
        a = left.get(key, NAN)
        b = right.get(key, NAN)
        pair = np.asarray([a, b], dtype=np.float64)
        if np.all(np.isnan(pair)):
            combined[f"node_{key}_min"] = NAN
            combined[f"node_{key}_max"] = NAN
            combined[f"node_{key}_mean"] = NAN
            continue
        combined[f"node_{key}_min"] = float(np.nanmin(pair))
        combined[f"node_{key}_max"] = float(np.nanmax(pair))
        combined[f"node_{key}_mean"] = float(np.nanmean(pair))
    return combined
