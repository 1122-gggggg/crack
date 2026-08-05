"""Skeleton graph extraction on synthetic shapes."""
from __future__ import annotations

import numpy as np

from conftest import build_graph
from topology_classifier.graph import NODE_ENDPOINT, NODE_JUNCTION, validate_graph


def _endpoints(graph) -> list:
    return [n for n in graph.nodes.values() if n.node_type == NODE_ENDPOINT]


def _junctions(graph) -> list:
    return [n for n in graph.nodes.values() if n.node_type == NODE_JUNCTION]


def test_straight_line(straight_line_mask, config):
    graph = build_graph(straight_line_mask, config)
    assert len(_endpoints(graph)) == 2
    assert len(_junctions(graph)) == 0
    assert len(graph.edges) == 1
    edge = next(iter(graph.edges.values()))
    assert edge.length_px() > 100
    assert validate_graph(graph).is_valid


def test_curved_line_tortuosity(curved_line_mask, config):
    graph = build_graph(curved_line_mask, config)
    assert len(_endpoints(graph)) == 2
    assert len(_junctions(graph)) == 0
    edge = max(graph.edges.values(), key=lambda e: e.length_px())
    euclid = float(np.hypot(*(edge.path[0].astype(float) - edge.path[-1].astype(float))))
    assert edge.length_px() / max(euclid, 1e-6) > 1.05


def test_y_junction(y_junction_mask, config):
    graph = build_graph(y_junction_mask, config)
    assert len(_endpoints(graph)) == 3
    assert len(_junctions(graph)) == 1
    long_edges = [e for e in graph.edges.values() if e.length_px() > 20]
    assert len(long_edges) == 3
    assert validate_graph(graph).is_valid


def test_x_junction(x_junction_mask, config):
    graph = build_graph(x_junction_mask, config)
    assert len(_endpoints(graph)) == 4
    assert len(_junctions(graph)) == 1
    long_edges = [e for e in graph.edges.values() if e.length_px() > 20]
    assert len(long_edges) == 4


def test_border_endpoint_flagged(border_line_mask, config):
    graph = build_graph(border_line_mask, config)
    endpoints = _endpoints(graph)
    assert len(endpoints) == 2
    assert sum(1 for n in endpoints if n.is_border_node) == 1
    summary = graph.summary()
    assert summary["valid_endpoint_count"] == 1


def test_empty_mask_is_safe(config):
    graph = build_graph(np.zeros((64, 64), dtype=np.uint8), config)
    assert not graph.nodes
    assert not graph.edges
    assert validate_graph(graph).is_valid


def test_single_pixel_component(config):
    mask = np.zeros((64, 64), dtype=np.uint8)
    mask[32, 32] = 1
    graph = build_graph(mask, config)
    assert len(graph.nodes) == 1
    assert not graph.edges


def test_coordinates_are_full_image(straight_line_mask, config):
    graph = build_graph(straight_line_mask, config)
    for edge in graph.edges.values():
        assert edge.path[:, 0].max() < straight_line_mask.shape[0]
        assert edge.path[:, 1].max() < straight_line_mask.shape[1]
        assert abs(float(np.mean(edge.path[:, 0])) - 100) < 3
