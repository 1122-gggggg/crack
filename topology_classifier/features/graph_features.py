"""Component-level and local-neighbourhood graph descriptors.

Classification happens per edge, but the discriminating evidence is contextual:
a straight segment inside a closed polygonal mesh is craquelure, the same
segment in isolation is a crack. These functions supply that context at two
scales -- the whole connected component and a bounded k-hop neighbourhood.
"""
from __future__ import annotations

import logging
from collections import deque
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set

import networkx as nx
import numpy as np

from ..config import GraphConfig
from ..graph.graph_types import NODE_ENDPOINT, NODE_JUNCTION, SkeletonEdge, SkeletonGraph
from .enclosed_faces import FaceMap
from .orientation import orientation_stats, segment_angles, segment_lengths

logger = logging.getLogger(__name__)

NAN = float("nan")


def cycle_rank(node_count: int, edge_count: int, component_count: int) -> int:
    """First Betti number ``E - V + C`` -- the number of independent cycles."""
    return int(edge_count - node_count + component_count)


def _length_stats(lengths: Sequence[float], prefix: str) -> Dict[str, float]:
    if not lengths:
        return {
            f"{prefix}_length_total": 0.0,
            f"{prefix}_length_mean": NAN,
            f"{prefix}_length_std": NAN,
            f"{prefix}_length_max": NAN,
            f"{prefix}_length_cv": NAN,
        }
    array = np.asarray(lengths, dtype=np.float64)
    mean = float(array.mean())
    std = float(array.std())
    return {
        f"{prefix}_length_total": float(array.sum()),
        f"{prefix}_length_mean": mean,
        f"{prefix}_length_std": std,
        f"{prefix}_length_max": float(array.max()),
        f"{prefix}_length_cv": float(std / mean) if mean > 1e-9 else NAN,
    }


def _orientation_summary(
    edges: Iterable[SkeletonEdge], bins: int, step: int, prefix: str
) -> Dict[str, float]:
    all_angles: List[np.ndarray] = []
    all_weights: List[np.ndarray] = []
    for edge in edges:
        angles = segment_angles(edge.path, step=step)
        weights = segment_lengths(edge.path, step=step)
        size = min(angles.size, weights.size)
        if size:
            all_angles.append(angles[:size])
            all_weights.append(weights[:size])
    if not all_angles:
        return orientation_stats(np.empty(0), bins=bins).as_dict(prefix)
    return orientation_stats(
        np.concatenate(all_angles), bins=bins, weights=np.concatenate(all_weights)
    ).as_dict(prefix)


def _structure_stats(
    graph: SkeletonGraph,
    node_ids: Set[int],
    edges: Sequence[SkeletonEdge],
    config: GraphConfig,
    prefix: str,
    component_count: int,
) -> Dict[str, float]:
    node_count = len(node_ids)
    edge_count = len(edges)
    degrees = [float(graph.nodes[n].degree) for n in node_ids if n in graph.nodes]
    endpoint_count = sum(
        1 for n in node_ids if n in graph.nodes and graph.nodes[n].node_type == NODE_ENDPOINT
    )
    junction_count = sum(
        1 for n in node_ids if n in graph.nodes and graph.nodes[n].node_type == NODE_JUNCTION
    )
    free_endpoint_count = sum(
        1
        for n in node_ids
        if n in graph.nodes
        and graph.nodes[n].node_type == NODE_ENDPOINT
        and not graph.nodes[n].is_border_node
    )
    rank = cycle_rank(node_count, edge_count, component_count)

    features: Dict[str, float] = {
        f"{prefix}_node_count": float(node_count),
        f"{prefix}_edge_count": float(edge_count),
        f"{prefix}_component_count": float(component_count),
        f"{prefix}_cycle_rank": float(rank),
        f"{prefix}_cycle_rank_per_edge": float(rank / edge_count) if edge_count else NAN,
        f"{prefix}_endpoint_count": float(endpoint_count),
        f"{prefix}_free_endpoint_count": float(free_endpoint_count),
        f"{prefix}_junction_count": float(junction_count),
        f"{prefix}_endpoint_ratio": float(endpoint_count / node_count) if node_count else NAN,
        f"{prefix}_junction_ratio": float(junction_count / node_count) if node_count else NAN,
        f"{prefix}_degree_mean": float(np.mean(degrees)) if degrees else NAN,
        f"{prefix}_degree_max": float(np.max(degrees)) if degrees else NAN,
        f"{prefix}_degree_std": float(np.std(degrees)) if degrees else NAN,
    }
    features.update(_length_stats([e.length_px() for e in edges], prefix))

    if node_ids:
        rows = np.asarray([graph.nodes[n].row for n in node_ids if n in graph.nodes], dtype=np.float64)
        cols = np.asarray([graph.nodes[n].col for n in node_ids if n in graph.nodes], dtype=np.float64)
        height = float(rows.max() - rows.min() + 1)
        width = float(cols.max() - cols.min() + 1)
        area = max(height * width, 1.0)
        features[f"{prefix}_bbox_height"] = height
        features[f"{prefix}_bbox_width"] = width
        features[f"{prefix}_bbox_diagonal"] = float(np.hypot(height, width))
        features[f"{prefix}_line_density"] = float(features[f"{prefix}_length_total"] / area)
    else:
        features[f"{prefix}_bbox_height"] = NAN
        features[f"{prefix}_bbox_width"] = NAN
        features[f"{prefix}_bbox_diagonal"] = NAN
        features[f"{prefix}_line_density"] = NAN

    features.update(
        _orientation_summary(edges, config.orientation_bins, config.curvature_resample_step, f"{prefix}_orient")
    )
    return features


def component_features(
    graph: SkeletonGraph,
    config: GraphConfig,
    strict_faces: Optional[FaceMap] = None,
    tolerant_faces: Optional[FaceMap] = None,
) -> Dict[int, Dict[str, float]]:
    """One feature dict per connected component, keyed by ``component_id``."""
    per_component: Dict[int, Dict[str, float]] = {}
    for component_id in graph.component_ids:
        edges = graph.edges_of_component(component_id)
        node_ids = {n.node_id for n in graph.nodes_of_component(component_id)}
        features = _structure_stats(graph, node_ids, edges, config, "comp", component_count=1)
        if strict_faces is not None:
            features["comp_strict_face_count"] = float(strict_faces.count)
            features["comp_strict_face_area_mean"] = (
                float(np.mean(strict_faces.area_list())) if strict_faces.count else 0.0
            )
        if tolerant_faces is not None:
            features["comp_tolerant_face_count"] = float(tolerant_faces.count)
            features["comp_face_closure_gain"] = float(
                tolerant_faces.count - (strict_faces.count if strict_faces else 0)
            )
        per_component[component_id] = features
    return per_component


@dataclass(frozen=True)
class LocalContext:
    node_ids: Set[int]
    edges: List[SkeletonEdge]
    component_count: int


def local_subgraph(
    graph: SkeletonGraph,
    edge: SkeletonEdge,
    hops: int,
    radius_pixels: float,
) -> LocalContext:
    """Edges within ``hops`` of the edge, bounded by a pixel radius.

    The radius keeps the neighbourhood local on very large panels, where a
    single component can span the entire image.
    """
    midpoint = edge.path[edge.path.shape[0] // 2].astype(np.float64)
    visited: Set[int] = set()
    queue: deque[tuple[int, int]] = deque()
    for node_id in (edge.u, edge.v):
        if node_id in graph.nodes:
            visited.add(node_id)
            queue.append((node_id, 0))

    while queue:
        node_id, depth = queue.popleft()
        if depth >= hops:
            continue
        for neighbour in graph.graph.neighbors(node_id):
            if neighbour in visited or neighbour not in graph.nodes:
                continue
            node = graph.nodes[neighbour]
            if float(np.hypot(node.row - midpoint[0], node.col - midpoint[1])) > radius_pixels:
                continue
            visited.add(neighbour)
            queue.append((neighbour, depth + 1))

    edges = [e for e in graph.edges.values() if e.u in visited and e.v in visited]
    subgraph = graph.graph.subgraph(visited)
    component_count = nx.number_connected_components(subgraph) if visited else 0
    return LocalContext(node_ids=visited, edges=edges, component_count=component_count)


def local_features(graph: SkeletonGraph, edge: SkeletonEdge, config: GraphConfig) -> Dict[str, float]:
    """Structure statistics of the bounded neighbourhood around one edge."""
    context = local_subgraph(graph, edge, config.local_graph_hops, float(config.local_radius_pixels))
    features = _structure_stats(
        graph, context.node_ids, context.edges, config, "local", component_count=context.component_count
    )
    own_length = edge.length_px()
    total = features.get("local_length_total", 0.0)
    features["local_length_share"] = float(own_length / total) if total > 1e-9 else NAN
    mean_length = features.get("local_length_mean", NAN)
    features["local_length_ratio"] = float(own_length / mean_length) if mean_length and mean_length > 1e-9 else NAN
    return features


def graph_summary_features(graph: SkeletonGraph, config: GraphConfig) -> Dict[str, float]:
    """Whole-image statistics, useful for reports and sanity checks."""
    node_ids = set(graph.nodes)
    edges = list(graph.edges.values())
    component_count = len(graph.component_ids)
    features = _structure_stats(graph, node_ids, edges, config, "image", component_count=component_count)
    height, width = graph.image_shape
    features["image_area"] = float(height * width)
    features["image_line_density"] = float(features["image_length_total"] / max(height * width, 1))
    return features
