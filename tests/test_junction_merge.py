"""Adjacent junction pixels must collapse into a single graph node."""
from __future__ import annotations

from conftest import build_graph
from topology_classifier.graph import NODE_JUNCTION, validate_graph


def test_junction_cluster_merges_to_single_node(junction_cluster_mask, config):
    graph = build_graph(junction_cluster_mask, config)
    junctions = [n for n in graph.nodes.values() if n.node_type == NODE_JUNCTION]
    assert len(junctions) == 1
    assert len(junctions[0].pixels) >= 2
    assert validate_graph(graph).is_valid


def test_junction_cluster_has_no_self_loop(junction_cluster_mask, config):
    graph = build_graph(junction_cluster_mask, config)
    assert all(not edge.is_loop for edge in graph.edges.values())


def test_merge_radius_zero_keeps_separate_nodes(junction_cluster_mask, config):
    from dataclasses import replace

    tight = replace(config.graph, junction_merge_radius=0)
    from topology_classifier.graph import SkeletonGraphBuilder, skeletonize_mask

    result = skeletonize_mask(junction_cluster_mask, connectivity=tight.connectivity)
    graph = SkeletonGraphBuilder(tight).build(result, image_id="synthetic")
    junctions = [n for n in graph.nodes.values() if n.node_type == NODE_JUNCTION]
    assert len(junctions) == 2
