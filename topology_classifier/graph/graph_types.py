"""Core data structures for skeleton graphs.

Coordinate convention
---------------------
All pixel coordinates are stored as ``(row, col)`` integer pairs in the frame of
the *full stitched image*. ``row`` grows downwards (image y), ``col`` grows to the
right (image x). Helper properties expose ``x``/``y`` for plotting code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import networkx as nx
import numpy as np

NodeType = str

NODE_ENDPOINT: NodeType = "endpoint"
NODE_JUNCTION: NodeType = "junction"
NODE_VIRTUAL_LOOP: NodeType = "virtual_loop"
NODE_ISOLATED: NodeType = "isolated"


@dataclass(frozen=True)
class SkeletonNode:
    """A graph node backed by one or more skeleton pixels."""

    node_id: int
    row: float
    col: float
    node_type: NodeType
    degree: int
    component_id: int
    is_border_node: bool = False
    pixels: Tuple[Tuple[int, int], ...] = ()

    @property
    def x(self) -> float:
        return self.col

    @property
    def y(self) -> float:
        return self.row


@dataclass(frozen=True)
class SkeletonEdge:
    """A skeleton path between two graph nodes.

    ``path`` is an ordered ``(N, 2)`` int32 array of ``(row, col)`` pixels that
    includes the pixels of the terminal nodes at both ends.
    """

    edge_id: int
    u: int
    v: int
    path: np.ndarray
    component_id: int
    is_loop: bool = False

    def __post_init__(self) -> None:
        if self.path.ndim != 2 or self.path.shape[1] != 2:
            raise ValueError(f"edge {self.edge_id}: path must be (N, 2), got {self.path.shape}")

    @property
    def rows(self) -> np.ndarray:
        return self.path[:, 0]

    @property
    def cols(self) -> np.ndarray:
        return self.path[:, 1]

    def length_px(self) -> float:
        if self.path.shape[0] < 2:
            return 0.0
        deltas = np.diff(self.path.astype(np.float64), axis=0)
        return float(np.sum(np.hypot(deltas[:, 0], deltas[:, 1])))


@dataclass
class SkeletonGraph:
    """Container holding nodes, edges and the backing :class:`networkx.MultiGraph`."""

    image_shape: Tuple[int, int]
    nodes: Dict[int, SkeletonNode] = field(default_factory=dict)
    edges: Dict[int, SkeletonEdge] = field(default_factory=dict)
    image_id: str = ""
    panel_id: str = ""
    graph: nx.MultiGraph = field(default_factory=nx.MultiGraph)
    component_of_node: Dict[int, int] = field(default_factory=dict)

    def add_node(self, node: SkeletonNode) -> None:
        self.nodes[node.node_id] = node
        self.component_of_node[node.node_id] = node.component_id
        self.graph.add_node(
            node.node_id,
            row=node.row,
            col=node.col,
            node_type=node.node_type,
            is_border_node=node.is_border_node,
            component_id=node.component_id,
        )

    def add_edge(self, edge: SkeletonEdge) -> None:
        self.edges[edge.edge_id] = edge
        self.graph.add_edge(
            edge.u,
            edge.v,
            key=edge.edge_id,
            edge_id=edge.edge_id,
            length=edge.length_px(),
            component_id=edge.component_id,
        )

    @property
    def component_ids(self) -> List[int]:
        return sorted({e.component_id for e in self.edges.values()} | {n.component_id for n in self.nodes.values()})

    def edges_of_component(self, component_id: int) -> List[SkeletonEdge]:
        return [e for e in self.edges.values() if e.component_id == component_id]

    def nodes_of_component(self, component_id: int) -> List[SkeletonNode]:
        return [n for n in self.nodes.values() if n.component_id == component_id]

    def incident_edges(self, node_id: int) -> List[SkeletonEdge]:
        found: List[SkeletonEdge] = []
        for edge in self.edges.values():
            if edge.u == node_id or edge.v == node_id:
                found.append(edge)
        return found

    def subgraph_of_component(self, component_id: int) -> nx.MultiGraph:
        node_ids = [n.node_id for n in self.nodes.values() if n.component_id == component_id]
        return self.graph.subgraph(node_ids).copy()

    def summary(self) -> Dict[str, float]:
        endpoints = [n for n in self.nodes.values() if n.node_type == NODE_ENDPOINT]
        junctions = [n for n in self.nodes.values() if n.node_type == NODE_JUNCTION]
        return {
            "node_count": float(len(self.nodes)),
            "edge_count": float(len(self.edges)),
            "endpoint_count": float(len(endpoints)),
            "valid_endpoint_count": float(sum(1 for n in endpoints if not n.is_border_node)),
            "junction_count": float(len(junctions)),
            "component_count": float(len(self.component_ids)),
            "skeleton_total_length": float(sum(e.length_px() for e in self.edges.values())),
        }

    def save(self, path: Path) -> None:
        """Persist the graph as a compressed ``.npz`` archive."""
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays: Dict[str, np.ndarray] = {}
        edge_meta = []
        for edge in self.edges.values():
            arrays[f"path_{edge.edge_id}"] = edge.path.astype(np.int32)
            edge_meta.append(
                {
                    "edge_id": edge.edge_id,
                    "u": edge.u,
                    "v": edge.v,
                    "component_id": edge.component_id,
                    "is_loop": edge.is_loop,
                }
            )
        node_meta = [
            {
                "node_id": n.node_id,
                "row": n.row,
                "col": n.col,
                "node_type": n.node_type,
                "degree": n.degree,
                "component_id": n.component_id,
                "is_border_node": n.is_border_node,
                "pixels": [list(p) for p in n.pixels],
            }
            for n in self.nodes.values()
        ]
        meta = {
            "image_shape": list(self.image_shape),
            "image_id": self.image_id,
            "panel_id": self.panel_id,
            "nodes": node_meta,
            "edges": edge_meta,
        }
        np.savez_compressed(path, meta=np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8), **arrays)

    @classmethod
    def load(cls, path: Path) -> "SkeletonGraph":
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(bytes(data["meta"]).decode("utf-8"))
            graph = cls(
                image_shape=tuple(meta["image_shape"]),  # type: ignore[arg-type]
                image_id=meta["image_id"],
                panel_id=meta["panel_id"],
            )
            for node in meta["nodes"]:
                graph.add_node(
                    SkeletonNode(
                        node_id=int(node["node_id"]),
                        row=float(node["row"]),
                        col=float(node["col"]),
                        node_type=str(node["node_type"]),
                        degree=int(node["degree"]),
                        component_id=int(node["component_id"]),
                        is_border_node=bool(node["is_border_node"]),
                        pixels=tuple((int(r), int(c)) for r, c in node["pixels"]),
                    )
                )
            for edge in meta["edges"]:
                graph.add_edge(
                    SkeletonEdge(
                        edge_id=int(edge["edge_id"]),
                        u=int(edge["u"]),
                        v=int(edge["v"]),
                        path=np.asarray(data[f"path_{edge['edge_id']}"], dtype=np.int32),
                        component_id=int(edge["component_id"]),
                        is_loop=bool(edge["is_loop"]),
                    )
                )
        return graph


def paths_to_pixel_index(edges: Iterable[SkeletonEdge], shape: Tuple[int, int]) -> np.ndarray:
    """Map every skeleton pixel to the id of the edge that owns it (-1 elsewhere)."""
    index = np.full(shape, -1, dtype=np.int32)
    for edge in edges:
        index[edge.rows, edge.cols] = edge.edge_id
    return index


def node_pixel_set(nodes: Iterable[SkeletonNode]) -> set[Tuple[int, int]]:
    pixels: set[Tuple[int, int]] = set()
    for node in nodes:
        pixels.update(node.pixels)
    return pixels


def optional_int(value: Optional[int], default: int) -> int:
    return default if value is None else int(value)
