"""Edge predictions must map back to pixels without changing the mask support."""
from __future__ import annotations

import numpy as np
import pytest
from conftest import build_graph

from topology_classifier.graph.graph_types import paths_to_pixel_index
from topology_classifier.inference.rasterize import (
    class_mask_to_color,
    nearest_edge_map,
    rasterize_predictions,
)


@pytest.fixture
def grid_graph(grid_mask, config):
    return build_graph(grid_mask, config), grid_mask.astype(bool)


def test_class_mask_support_equals_input_mask(grid_graph, config):
    graph, mask = grid_graph
    labels = {edge_id: "crack" for edge_id in graph.edges}
    result = rasterize_predictions(graph, mask, labels, config.classes)

    assert result.class_mask.shape == mask.shape
    assigned = result.class_mask != config.classes.background
    assert np.array_equal(assigned, mask)
    assert result.pixel_counts[config.classes.crack] == int(mask.sum())


def test_two_classes_partition_the_foreground(grid_graph, config):
    graph, mask = grid_graph
    edge_ids = sorted(graph.edges)
    labels = {
        edge_id: ("crack" if index % 2 == 0 else "craquelure")
        for index, edge_id in enumerate(edge_ids)
    }
    result = rasterize_predictions(graph, mask, labels, config.classes)

    crack = result.pixel_counts.get(config.classes.crack, 0)
    craquelure = result.pixel_counts.get(config.classes.craquelure, 0)
    assert crack > 0 and craquelure > 0
    assert crack + craquelure == int(mask.sum())


def test_uncertain_edges_become_ignore_not_a_forced_class(grid_graph, config):
    graph, mask = grid_graph
    labels = {edge_id: "uncertain" for edge_id in graph.edges}
    result = rasterize_predictions(graph, mask, labels, config.classes)

    assert result.pixel_counts.get(config.classes.ignore, 0) == int(mask.sum())
    assert config.classes.crack not in result.pixel_counts
    assert config.classes.craquelure not in result.pixel_counts


def test_unlabeled_edges_become_ignore(grid_graph, config):
    graph, mask = grid_graph
    kept = sorted(graph.edges)[0]
    result = rasterize_predictions(graph, mask, {kept: "crack"}, config.classes)

    assert result.pixel_counts.get(config.classes.crack, 0) > 0
    assert result.pixel_counts.get(config.classes.ignore, 0) > 0


def test_skeleton_pixels_keep_their_own_edge_id(grid_graph):
    graph, mask = grid_graph
    edge_index = paths_to_pixel_index(graph.edges.values(), graph.image_shape)
    propagated = nearest_edge_map(edge_index, mask)

    seeds = edge_index >= 0
    assert seeds.any()
    assert np.array_equal(propagated[seeds], edge_index[seeds])


def test_propagation_agrees_with_the_exact_distance_transform(grid_graph):
    from scipy import ndimage as ndi

    graph, mask = grid_graph
    edge_index = paths_to_pixel_index(graph.edges.values(), graph.image_shape)
    propagated = nearest_edge_map(edge_index, mask)

    _, indices = ndi.distance_transform_edt(edge_index < 0, return_indices=True)
    reference = edge_index[indices[0], indices[1]]
    reference[~mask] = -1

    agreement = float((propagated[mask] == reference[mask]).mean())
    assert agreement > 0.90  # cv2 uses a 5x5 approximation, so ties may differ
    assert set(np.unique(propagated[mask])).issubset(set(graph.edges))


def test_background_stays_unassigned(grid_graph):
    graph, mask = grid_graph
    edge_index = paths_to_pixel_index(graph.edges.values(), graph.image_shape)
    propagated = nearest_edge_map(edge_index, mask)

    assert (propagated[~mask] == -1).all()


def test_empty_skeleton_yields_no_assignment(config):
    empty = np.zeros((32, 32), dtype=np.int32) - 1
    mask = np.ones((32, 32), dtype=bool)

    propagated = nearest_edge_map(empty, mask)

    assert (propagated == -1).all()


def test_color_map_is_bgr_and_matches_class_values(grid_graph, config):
    graph, mask = grid_graph
    labels = {edge_id: "crack" for edge_id in graph.edges}
    result = rasterize_predictions(graph, mask, labels, config.classes)

    color = class_mask_to_color(result.class_mask, config.classes)

    assert color.shape == (*mask.shape, 3)
    assert color.dtype == np.uint8
    assert (color[~mask] == 0).all()
    assert not (color[mask] == 0).all()
