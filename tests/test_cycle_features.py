"""Cycle rank and local-context features on known topologies."""
from __future__ import annotations

import numpy as np

from conftest import build_graph
from topology_classifier.features import component_features, cycle_rank, graph_summary_features, local_subgraph


def test_cycle_rank_formula():
    assert cycle_rank(node_count=4, edge_count=4, component_count=1) == 1
    assert cycle_rank(node_count=4, edge_count=3, component_count=1) == 0
    assert cycle_rank(node_count=6, edge_count=4, component_count=2) == 0


def test_tree_has_zero_cycle_rank(y_junction_mask, config):
    graph = build_graph(y_junction_mask, config)
    summary = graph_summary_features(graph, config.graph)
    assert summary["image_cycle_rank"] == 0.0
    assert summary["image_junction_count"] == 1.0
    assert summary["image_endpoint_count"] == 3.0


def test_ring_has_unit_cycle_rank(closed_loop_mask, config):
    graph = build_graph(closed_loop_mask, config)
    summary = graph_summary_features(graph, config.graph)
    assert summary["image_cycle_rank"] == 1.0
    assert summary["image_endpoint_count"] == 0.0


def test_grid_cycle_rank_matches_cell_count(grid_mask, config):
    graph = build_graph(grid_mask, config)
    summary = graph_summary_features(graph, config.graph)
    assert summary["image_cycle_rank"] >= 9.0
    assert summary["image_junction_ratio"] > 0.4


def test_component_features_are_per_component(config):
    from conftest import draw_line

    mask = np.zeros((200, 200), dtype=np.uint8)
    draw_line(mask, (40, 20), (40, 180))  # open line
    draw_line(mask, (120, 20), (120, 100))  # second component, forms a triangle
    draw_line(mask, (120, 100), (170, 20))
    draw_line(mask, (170, 20), (120, 20))
    graph = build_graph(mask, config)
    per_component = component_features(graph, config.graph)
    assert len(per_component) == 2
    ranks = sorted(f["comp_cycle_rank"] for f in per_component.values())
    assert ranks == [0.0, 1.0]


def test_local_subgraph_is_bounded_by_radius(grid_mask, config):
    from dataclasses import replace

    graph = build_graph(grid_mask, config)
    edge = max(graph.edges.values(), key=lambda e: e.length_px())
    wide = local_subgraph(graph, edge, hops=2, radius_pixels=10_000)
    narrow = local_subgraph(graph, edge, hops=2, radius_pixels=20)
    assert len(narrow.node_ids) <= len(wide.node_ids)
    assert len(wide.node_ids) > len(narrow.node_ids)
    tight_config = replace(config.graph, local_graph_hops=1)
    one_hop = local_subgraph(graph, edge, hops=tight_config.local_graph_hops, radius_pixels=10_000)
    assert len(one_hop.node_ids) <= len(wide.node_ids)
