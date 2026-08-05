"""Orientation statistics must be direction-invariant and scale-sensible."""
from __future__ import annotations

import numpy as np
import pytest

from conftest import build_graph
from topology_classifier.features import orientation_stats, path_orientation_stats, segment_angles
from topology_classifier.features.orientation import angular_difference_deg

BINS = 12


def _path(rows, cols) -> np.ndarray:
    return np.stack([np.asarray(rows), np.asarray(cols)], axis=1).astype(np.int32)


def test_horizontal_line_is_maximally_anisotropic():
    path = _path(np.full(50, 10), np.arange(50))
    stats = path_orientation_stats(path, bins=BINS)
    assert stats.anisotropy == pytest.approx(1.0, abs=1e-6)
    assert stats.circular_variance == pytest.approx(0.0, abs=1e-6)
    assert stats.entropy == pytest.approx(0.0, abs=1e-6)
    assert stats.dominant_deg == pytest.approx(0.0, abs=1e-6)


def test_orientation_is_invariant_to_path_direction():
    path = _path(np.arange(40), np.arange(40) * 2)
    forward = path_orientation_stats(path, bins=BINS)
    backward = path_orientation_stats(path[::-1].copy(), bins=BINS)
    assert forward.dominant_deg == pytest.approx(backward.dominant_deg, abs=1e-6)
    assert forward.anisotropy == pytest.approx(backward.anisotropy, abs=1e-6)
    assert np.allclose(forward.histogram, backward.histogram[::-1] if False else forward.histogram)


def test_isotropic_angles_have_low_anisotropy_and_high_entropy():
    angles = np.linspace(0.0, np.pi, 360, endpoint=False)
    stats = orientation_stats(angles, bins=BINS)
    assert stats.anisotropy < 1e-6
    assert stats.entropy > 0.99


def test_empty_path_returns_nan_without_raising():
    stats = path_orientation_stats(np.empty((0, 2), dtype=np.int32), bins=BINS)
    assert np.isnan(stats.dominant_deg)
    assert stats.histogram.shape == (BINS,)


def test_angular_difference_wraps_at_180():
    assert angular_difference_deg(10.0, 170.0) == pytest.approx(20.0)
    assert angular_difference_deg(0.0, 180.0) == pytest.approx(0.0)
    assert angular_difference_deg(0.0, 90.0) == pytest.approx(90.0)


def test_grid_network_is_more_isotropic_than_single_line(grid_mask, straight_line_mask, config):
    grid_graph = build_graph(grid_mask, config)
    line_graph = build_graph(straight_line_mask, config)
    grid_angles = np.concatenate([segment_angles(e.path, 5) for e in grid_graph.edges.values() if e.path.shape[0] > 6])
    line_angles = np.concatenate([segment_angles(e.path, 5) for e in line_graph.edges.values()])
    grid_stats = orientation_stats(grid_angles, bins=BINS)
    line_stats = orientation_stats(line_angles, bins=BINS)
    assert grid_stats.entropy > line_stats.entropy
    assert grid_stats.anisotropy < line_stats.anisotropy
