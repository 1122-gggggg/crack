"""Strict vs tolerant enclosed-face detection."""
from __future__ import annotations

import numpy as np

from conftest import build_graph, draw_line, draw_ring
from topology_classifier.features import edge_face_adjacency, enclosed_faces


def test_open_line_has_no_face(straight_line_mask, config):
    faces = enclosed_faces(straight_line_mask.astype(bool), config.graph)
    assert faces.count == 0


def test_ring_encloses_one_face(closed_loop_mask, config):
    faces = enclosed_faces(closed_loop_mask.astype(bool), config.graph)
    assert faces.count == 1
    assert faces.total_area > 5000  # radius 55 disc minus the ring itself


def test_grid_encloses_interior_cells(grid_mask, config):
    faces = enclosed_faces(grid_mask.astype(bool), config.graph)
    assert faces.count >= 9
    assert min(faces.area_list()) >= config.graph.minimum_enclosed_area


def test_minimum_area_filters_pinholes(config):
    from dataclasses import replace

    mask = np.zeros((60, 60), dtype=bool)
    mask[20:24, 20:24] = True
    mask[21:23, 21:23] = False  # 2x2 = 4 px hole
    lenient = replace(config.graph, minimum_enclosed_area=1)
    strict = replace(config.graph, minimum_enclosed_area=16)
    assert enclosed_faces(mask, lenient).count == 1
    assert enclosed_faces(mask, strict).count == 0


def test_tolerant_closing_recovers_a_broken_cell(config):
    """A 2 px break is bridged by radius-1 closing; the strict pass still sees it."""
    mask = np.zeros((120, 120), dtype=np.uint8)
    draw_ring(mask, (60, 60), 35, thickness=3)
    mask[59:61, 88:104] = 0  # 2 px break across the ring, within reach of disk(1)
    binary = mask.astype(bool)
    strict = enclosed_faces(binary, config.graph, tolerant=False)
    tolerant = enclosed_faces(binary, config.graph, tolerant=True)
    assert strict.count == 0
    assert tolerant.count == 1


def test_tolerant_closing_does_not_bridge_wide_gaps(config):
    """Closing must stay conservative: a 6 px break stays open in both passes."""
    mask = np.zeros((120, 120), dtype=np.uint8)
    draw_ring(mask, (60, 60), 35, thickness=3)
    mask[57:63, 88:104] = 0
    binary = mask.astype(bool)
    assert enclosed_faces(binary, config.graph, tolerant=False).count == 0
    assert enclosed_faces(binary, config.graph, tolerant=True).count == 0


def test_edge_face_adjacency_marks_the_ring_edge(closed_loop_mask, config):
    graph = build_graph(closed_loop_mask, config)
    faces = enclosed_faces(closed_loop_mask.astype(bool), config.graph)
    paths = {edge_id: edge.path for edge_id, edge in graph.edges.items()}
    adjacency = edge_face_adjacency(paths, faces, search_radius=3)
    assert adjacency
    assert all(item.borders_face for item in adjacency.values())
    assert all(item.face_count == 1 for item in adjacency.values())


def test_edge_face_adjacency_is_empty_for_open_lines(straight_line_mask, config):
    graph = build_graph(straight_line_mask, config)
    faces = enclosed_faces(straight_line_mask.astype(bool), config.graph)
    paths = {edge_id: edge.path for edge_id, edge in graph.edges.items()}
    adjacency = edge_face_adjacency(paths, faces, search_radius=3)
    assert all(not item.borders_face for item in adjacency.values())
