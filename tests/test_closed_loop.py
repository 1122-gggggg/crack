"""Closed rings and grid networks: cycle structure must survive graph building."""
from __future__ import annotations

import networkx as nx

from conftest import build_graph
from topology_classifier.graph import NODE_ENDPOINT, NODE_JUNCTION, NODE_VIRTUAL_LOOP, validate_graph


def _cycle_rank(graph) -> int:
    edges = len(graph.edges)
    nodes = len(graph.nodes)
    components = nx.number_connected_components(graph.graph) if nodes else 0
    return edges - nodes + components


def test_closed_loop_has_no_endpoint(closed_loop_mask, config):
    graph = build_graph(closed_loop_mask, config)
    assert not [n for n in graph.nodes.values() if n.node_type == NODE_ENDPOINT]
    assert [n for n in graph.nodes.values() if n.node_type == NODE_VIRTUAL_LOOP]
    assert validate_graph(graph).is_valid


def test_closed_loop_cycle_rank_is_one(closed_loop_mask, config):
    graph = build_graph(closed_loop_mask, config)
    assert _cycle_rank(graph) == 1


def test_closed_loop_edge_is_marked_as_loop(closed_loop_mask, config):
    graph = build_graph(closed_loop_mask, config)
    loops = [e for e in graph.edges.values() if e.is_loop]
    assert len(loops) == 1
    path = loops[0].path
    assert tuple(path[0]) == tuple(path[-1])
    assert loops[0].length_px() > 300  # circumference of a radius-55 ring


def test_grid_network_has_many_junctions_and_cycles(grid_mask, config):
    graph = build_graph(grid_mask, config)
    junctions = [n for n in graph.nodes.values() if n.node_type == NODE_JUNCTION]
    assert len(junctions) >= 16  # 5 x 5 interior crossings, some merged
    assert _cycle_rank(graph) >= 9  # 4 x 4 enclosed cells at minimum
    assert validate_graph(graph).is_valid


def test_grid_network_is_one_component(grid_mask, config):
    graph = build_graph(grid_mask, config)
    assert len(graph.component_ids) == 1
