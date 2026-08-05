"""Build a :class:`SkeletonGraph` from a binary skeleton.

The builder handles the four cases that naive pixel-graph extraction gets wrong:

* junction clusters -- several adjacent ``degree >= 3`` pixels collapse into one node,
* pure closed loops -- components without endpoint/junction get a virtual node,
* border endpoints -- endpoints near the image edge are flagged, not counted as real,
* parallel edges -- two nodes may be joined by several distinct skeleton paths.
"""
from __future__ import annotations

import logging
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import disk

from ..config import GraphConfig
from .graph_types import (
    NODE_ENDPOINT,
    NODE_ISOLATED,
    NODE_JUNCTION,
    NODE_VIRTUAL_LOOP,
    SkeletonEdge,
    SkeletonGraph,
    SkeletonNode,
)
from .skeletonize import SkeletonResult, structure_for

logger = logging.getLogger(__name__)

Pixel = Tuple[int, int]

_OFFSETS: Tuple[Pixel, ...] = (
    (-1, -1), (-1, 0), (-1, 1),
    (0, -1), (0, 1),
    (1, -1), (1, 0), (1, 1),
)


def _neighbors(pixel: Pixel, shape: Tuple[int, int]) -> Iterable[Pixel]:
    row, col = pixel
    height, width = shape
    for d_row, d_col in _OFFSETS:
        r, c = row + d_row, col + d_col
        if 0 <= r < height and 0 <= c < width:
            yield r, c


def _region_pixels(labels: np.ndarray, boxes: Sequence[Optional[Tuple[slice, slice]]], index: int) -> List[Pixel]:
    """Pixels of one labelled region, read through its bounding box.

    Comparing the whole label image against every index is quadratic in the
    number of regions; on a 9147x4374 panel with thousands of segments that
    alone costs minutes. ``find_objects`` keeps each lookup local.
    """
    box = boxes[index - 1]
    if box is None:
        return []
    rows, cols = np.nonzero(labels[box] == index)
    return list(zip((rows + box[0].start).tolist(), (cols + box[1].start).tolist()))


class SkeletonGraphBuilder:
    """Convert a skeleton image into a node/edge graph in full-image coordinates."""

    def __init__(self, config: GraphConfig) -> None:
        self.config = config

    def build(
        self,
        result: SkeletonResult,
        *,
        image_id: str = "",
        panel_id: str = "",
        valid_region: Optional[np.ndarray] = None,
    ) -> SkeletonGraph:
        skeleton = result.skeleton
        shape: Tuple[int, int] = skeleton.shape  # type: ignore[assignment]
        graph = SkeletonGraph(image_shape=shape, image_id=image_id, panel_id=panel_id)
        if not skeleton.any():
            logger.warning("[%s] empty skeleton, returning empty graph", image_id)
            return graph

        degree = result.degree
        components = result.component_labels
        node_pixel_map = np.full(shape, -1, dtype=np.int32)
        next_node_id = 0

        junction_clusters = self._junction_clusters(skeleton, degree)
        for pixels in junction_clusters:
            node_id = next_node_id
            next_node_id += 1
            self._register_node(
                graph,
                node_pixel_map,
                node_id,
                pixels,
                NODE_JUNCTION,
                components,
                shape,
                valid_region,
                degree_hint=None,
            )

        endpoint_pixels = list(zip(*np.nonzero(skeleton & (degree == 1))))
        for pixel in endpoint_pixels:
            if node_pixel_map[pixel] >= 0:
                continue
            node_id = next_node_id
            next_node_id += 1
            self._register_node(
                graph, node_pixel_map, node_id, [pixel], NODE_ENDPOINT, components, shape, valid_region, degree_hint=1
            )

        isolated_pixels = list(zip(*np.nonzero(skeleton & (degree == 0))))
        for pixel in isolated_pixels:
            if node_pixel_map[pixel] >= 0:
                continue
            node_id = next_node_id
            next_node_id += 1
            self._register_node(
                graph, node_pixel_map, node_id, [pixel], NODE_ISOLATED, components, shape, valid_region, degree_hint=0
            )

        next_edge_id = 0
        next_edge_id = self._add_path_edges(graph, result, node_pixel_map, next_edge_id)
        next_edge_id = self._add_adjacent_node_edges(graph, node_pixel_map, components, next_edge_id)
        next_edge_id = self._add_pure_loop_edges(graph, result, node_pixel_map, next_edge_id, valid_region)

        self._refresh_node_degrees(graph)
        logger.info(
            "[%s] graph: %d nodes, %d edges, %d components",
            image_id,
            len(graph.nodes),
            len(graph.edges),
            len(graph.component_ids),
        )
        return graph

    # ------------------------------------------------------------------ nodes

    def _junction_clusters(self, skeleton: np.ndarray, degree: np.ndarray) -> List[List[Pixel]]:
        """Group ``degree >= 3`` pixels (and the skeleton pixels bridging them)."""
        junction_mask = skeleton & (degree >= 3)
        if not junction_mask.any():
            return []

        structure = structure_for(8)
        raw_labels, raw_count = ndi.label(junction_mask, structure=structure)
        raw_boxes = ndi.find_objects(raw_labels)
        radius = max(0, int(self.config.junction_merge_radius))
        if radius == 0:
            return [_region_pixels(raw_labels, raw_boxes, idx) for idx in range(1, raw_count + 1)]

        dilated = ndi.binary_dilation(junction_mask, structure=disk(radius).astype(bool))
        region_labels, region_count = ndi.label(dilated, structure=structure)
        region_boxes = ndi.find_objects(region_labels)

        clusters: List[List[Pixel]] = []
        for region_id in range(1, region_count + 1):
            box = region_boxes[region_id - 1]
            if box is None:
                continue
            region = region_labels[box] == region_id
            local_junctions = junction_mask[box]
            raw_in_region = np.unique(raw_labels[box][region & local_junctions])
            raw_in_region = raw_in_region[raw_in_region > 0]
            if raw_in_region.size == 0:
                continue
            if raw_in_region.size == 1:
                clusters.append(_region_pixels(raw_labels, raw_boxes, int(raw_in_region[0])))
                continue
            # Several junction blobs sit within the merge radius: absorb the
            # skeleton pixels bridging them so no spurious self-loop is created.
            bridge = skeleton[box] & region
            bridge_labels, bridge_count = ndi.label(bridge, structure=structure)
            row0, col0 = box[0].start, box[1].start
            for bridge_id in range(1, bridge_count + 1):
                piece = bridge_labels == bridge_id
                if not (piece & local_junctions).any():
                    continue
                rows, cols = np.nonzero(piece)
                clusters.append(list(zip((rows + row0).tolist(), (cols + col0).tolist())))
        return clusters

    def _register_node(
        self,
        graph: SkeletonGraph,
        node_pixel_map: np.ndarray,
        node_id: int,
        pixels: Sequence[Pixel],
        node_type: str,
        components: np.ndarray,
        shape: Tuple[int, int],
        valid_region: Optional[np.ndarray],
        degree_hint: Optional[int],
    ) -> None:
        arr = np.asarray(pixels, dtype=np.int32).reshape(-1, 2)
        centroid = arr.mean(axis=0)
        distances = np.hypot(arr[:, 0] - centroid[0], arr[:, 1] - centroid[1])
        medoid = arr[int(np.argmin(distances))]
        component_id = int(components[medoid[0], medoid[1]])
        node = SkeletonNode(
            node_id=node_id,
            row=float(medoid[0]),
            col=float(medoid[1]),
            node_type=node_type,
            degree=0 if degree_hint is None else degree_hint,
            component_id=component_id,
            is_border_node=self._is_border(arr, shape, valid_region),
            pixels=tuple((int(r), int(c)) for r, c in arr),
        )
        graph.add_node(node)
        node_pixel_map[arr[:, 0], arr[:, 1]] = node_id

    def _is_border(self, pixels: np.ndarray, shape: Tuple[int, int], valid_region: Optional[np.ndarray]) -> bool:
        margin = int(self.config.border_margin)
        height, width = shape
        near_image_border = bool(
            np.any(pixels[:, 0] < margin)
            or np.any(pixels[:, 1] < margin)
            or np.any(pixels[:, 0] >= height - margin)
            or np.any(pixels[:, 1] >= width - margin)
        )
        if near_image_border or valid_region is None or margin <= 0:
            return near_image_border
        eroded = ndi.binary_erosion(valid_region, structure=disk(margin).astype(bool), border_value=0)
        return bool(np.any(~eroded[pixels[:, 0], pixels[:, 1]]))

    def _refresh_node_degrees(self, graph: SkeletonGraph) -> None:
        counts: Dict[int, int] = {node_id: 0 for node_id in graph.nodes}
        for edge in graph.edges.values():
            counts[edge.u] = counts.get(edge.u, 0) + 1
            counts[edge.v] = counts.get(edge.v, 0) + 1
        for node_id, node in list(graph.nodes.items()):
            graph.nodes[node_id] = SkeletonNode(
                node_id=node.node_id,
                row=node.row,
                col=node.col,
                node_type=node.node_type,
                degree=counts.get(node_id, 0),
                component_id=node.component_id,
                is_border_node=node.is_border_node,
                pixels=node.pixels,
            )
            graph.graph.nodes[node_id]["degree"] = counts.get(node_id, 0)

    # ------------------------------------------------------------------ edges

    def _add_path_edges(
        self,
        graph: SkeletonGraph,
        result: SkeletonResult,
        node_pixel_map: np.ndarray,
        next_edge_id: int,
    ) -> int:
        shape: Tuple[int, int] = result.skeleton.shape  # type: ignore[assignment]
        path_mask = result.skeleton & (node_pixel_map < 0)
        if not path_mask.any():
            return next_edge_id

        labels, count = ndi.label(path_mask, structure=structure_for(8))
        boxes = ndi.find_objects(labels)
        for segment_id in range(1, count + 1):
            pixels = set(_region_pixels(labels, boxes, segment_id))
            ordered = _trace_chain(pixels, shape)
            if not ordered:
                continue
            head_nodes = _adjacent_nodes(ordered[0], node_pixel_map, shape)
            tail_nodes = _adjacent_nodes(ordered[-1], node_pixel_map, shape)
            if not head_nodes or not tail_nodes:
                # Segment dangling on one side: keep it as a self-standing edge
                # anchored on whichever node exists (or skip when fully isolated).
                if not head_nodes and not tail_nodes:
                    continue
                anchors = head_nodes or tail_nodes
                u = v = anchors[0]
            else:
                u = head_nodes[0]
                v = tail_nodes[0]

            head_pixel = _closest_node_pixel(graph, u, ordered[0])
            tail_pixel = _closest_node_pixel(graph, v, ordered[-1])
            path = np.asarray([head_pixel, *ordered, tail_pixel], dtype=np.int32)
            component_id = int(result.component_labels[ordered[0]])
            graph.add_edge(
                SkeletonEdge(
                    edge_id=next_edge_id,
                    u=u,
                    v=v,
                    path=path,
                    component_id=component_id,
                    is_loop=u == v,
                )
            )
            next_edge_id += 1
        return next_edge_id

    def _add_adjacent_node_edges(
        self,
        graph: SkeletonGraph,
        node_pixel_map: np.ndarray,
        components: np.ndarray,
        next_edge_id: int,
    ) -> int:
        """Connect node clusters that touch directly, without an intermediate path."""
        shape: Tuple[int, int] = node_pixel_map.shape  # type: ignore[assignment]
        seen: Set[Tuple[int, int]] = set()
        for node in list(graph.nodes.values()):
            for pixel in node.pixels:
                for neighbor in _neighbors(pixel, shape):
                    other = int(node_pixel_map[neighbor])
                    if other < 0 or other == node.node_id:
                        continue
                    key = (min(node.node_id, other), max(node.node_id, other))
                    if key in seen:
                        continue
                    seen.add(key)
                    path = np.asarray([pixel, neighbor], dtype=np.int32)
                    graph.add_edge(
                        SkeletonEdge(
                            edge_id=next_edge_id,
                            u=key[0],
                            v=key[1],
                            path=path,
                            component_id=int(components[pixel]),
                            is_loop=False,
                        )
                    )
                    next_edge_id += 1
        return next_edge_id

    def _add_pure_loop_edges(
        self,
        graph: SkeletonGraph,
        result: SkeletonResult,
        node_pixel_map: np.ndarray,
        next_edge_id: int,
        valid_region: Optional[np.ndarray],
    ) -> int:
        """Handle components made only of ``degree == 2`` pixels (closed rings)."""
        shape: Tuple[int, int] = result.skeleton.shape  # type: ignore[assignment]
        remaining = result.skeleton & (node_pixel_map < 0)
        covered: Set[Pixel] = set()
        for edge in graph.edges.values():
            covered.update((int(r), int(c)) for r, c in edge.path)

        labels, count = ndi.label(remaining, structure=structure_for(8))
        boxes = ndi.find_objects(labels)
        next_node_id = (max(graph.nodes) + 1) if graph.nodes else 0
        for segment_id in range(1, count + 1):
            pixels = set(_region_pixels(labels, boxes, segment_id))
            if pixels & covered:
                continue
            ring = _trace_ring(pixels, shape)
            if len(ring) < 3:
                continue
            anchor = ring[0]
            node_id = next_node_id
            next_node_id += 1
            self._register_node(
                graph,
                node_pixel_map,
                node_id,
                [anchor],
                NODE_VIRTUAL_LOOP,
                result.component_labels,
                shape,
                valid_region,
                degree_hint=2,
            )
            path = np.asarray([*ring, anchor], dtype=np.int32)
            graph.add_edge(
                SkeletonEdge(
                    edge_id=next_edge_id,
                    u=node_id,
                    v=node_id,
                    path=path,
                    component_id=int(result.component_labels[anchor]),
                    is_loop=True,
                )
            )
            next_edge_id += 1
        return next_edge_id


# ---------------------------------------------------------------- tracing utils


def _trace_chain(pixels: Set[Pixel], shape: Tuple[int, int]) -> List[Pixel]:
    """Return the pixels of a thin segment ordered from one end to the other."""
    if not pixels:
        return []
    if len(pixels) == 1:
        return [next(iter(pixels))]

    ends = [p for p in pixels if sum(1 for n in _neighbors(p, shape) if n in pixels) <= 1]
    start = min(ends) if ends else min(pixels)

    ordered: List[Pixel] = [start]
    visited: Set[Pixel] = {start}
    while True:
        current = ordered[-1]
        candidates = [n for n in _neighbors(current, shape) if n in pixels and n not in visited]
        if not candidates:
            break
        # Prefer 4-connected continuation to keep the path geometrically tight.
        candidates.sort(key=lambda p: (abs(p[0] - current[0]) + abs(p[1] - current[1]), p))
        nxt = candidates[0]
        ordered.append(nxt)
        visited.add(nxt)
    return ordered


def _trace_ring(pixels: Set[Pixel], shape: Tuple[int, int]) -> List[Pixel]:
    ordered = _trace_chain(pixels, shape)
    return ordered


def _adjacent_nodes(pixel: Pixel, node_pixel_map: np.ndarray, shape: Tuple[int, int]) -> List[int]:
    found: List[int] = []
    for neighbor in _neighbors(pixel, shape):
        node_id = int(node_pixel_map[neighbor])
        if node_id >= 0 and node_id not in found:
            found.append(node_id)
    return found


def _closest_node_pixel(graph: SkeletonGraph, node_id: int, reference: Pixel) -> Pixel:
    node = graph.nodes[node_id]
    best = min(node.pixels, key=lambda p: (p[0] - reference[0]) ** 2 + (p[1] - reference[1]) ** 2)
    return int(best[0]), int(best[1])
