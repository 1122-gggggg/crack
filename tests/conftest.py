"""Synthetic mask fixtures shared by the topology tests."""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Tuple

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from topology_classifier.config import TopologyConfig, load_config  # noqa: E402
from topology_classifier.graph import SkeletonGraph, SkeletonGraphBuilder, skeletonize_mask  # noqa: E402

CANVAS: Tuple[int, int] = (200, 200)


def _blank(shape: Tuple[int, int] = CANVAS) -> np.ndarray:
    return np.zeros(shape, dtype=np.uint8)


def draw_line(mask: np.ndarray, start: Tuple[int, int], end: Tuple[int, int], thickness: int = 3) -> np.ndarray:
    """Draw a thick line between ``(row, col)`` points using pure numpy."""
    (r0, c0), (r1, c1) = start, end
    steps = int(max(abs(r1 - r0), abs(c1 - c0))) + 1
    rows = np.linspace(r0, r1, steps * 2)
    cols = np.linspace(c0, c1, steps * 2)
    half = thickness // 2
    for r, c in zip(rows, cols):
        rr = int(round(r))
        cc = int(round(c))
        mask[
            max(0, rr - half) : min(mask.shape[0], rr + half + 1),
            max(0, cc - half) : min(mask.shape[1], cc + half + 1),
        ] = 1
    return mask


def draw_ring(mask: np.ndarray, center: Tuple[int, int], radius: int, thickness: int = 3) -> np.ndarray:
    rows, cols = np.ogrid[: mask.shape[0], : mask.shape[1]]
    dist = np.hypot(rows - center[0], cols - center[1])
    mask[(dist >= radius - thickness / 2) & (dist <= radius + thickness / 2)] = 1
    return mask


@pytest.fixture(scope="session")
def config() -> TopologyConfig:
    return load_config()


@pytest.fixture
def straight_line_mask() -> np.ndarray:
    return draw_line(_blank(), (100, 30), (100, 170))


@pytest.fixture
def curved_line_mask() -> np.ndarray:
    mask = _blank()
    cols = np.arange(30, 171)
    rows = (100 + 35 * np.sin((cols - 30) / 140 * np.pi)).astype(int)
    for r, c in zip(rows, cols):
        mask[r - 1 : r + 2, c - 1 : c + 2] = 1
    return mask


@pytest.fixture
def y_junction_mask() -> np.ndarray:
    mask = _blank()
    draw_line(mask, (100, 100), (100, 30))
    draw_line(mask, (100, 100), (40, 170))
    draw_line(mask, (100, 100), (160, 170))
    return mask


@pytest.fixture
def x_junction_mask() -> np.ndarray:
    mask = _blank()
    draw_line(mask, (30, 30), (170, 170))
    draw_line(mask, (170, 30), (30, 170))
    return mask


@pytest.fixture
def closed_loop_mask() -> np.ndarray:
    return draw_ring(_blank(), (100, 100), 55)


@pytest.fixture
def grid_mask() -> np.ndarray:
    """Craquelure-like network: several crossing lines forming closed cells."""
    mask = _blank()
    for row in range(40, 171, 32):
        draw_line(mask, (row, 30), (row, 170), thickness=3)
    for col in range(40, 171, 32):
        draw_line(mask, (30, col), (170, col), thickness=3)
    return mask


@pytest.fixture
def border_line_mask() -> np.ndarray:
    """Line that runs off the left image border."""
    return draw_line(_blank(), (100, 0), (100, 120))


@pytest.fixture
def junction_cluster_mask() -> np.ndarray:
    """Two T-junctions three pixels apart, i.e. one real crossing."""
    mask = _blank()
    draw_line(mask, (30, 100), (170, 100), thickness=3)
    draw_line(mask, (99, 100), (99, 40), thickness=1)
    draw_line(mask, (102, 100), (102, 160), thickness=1)
    return mask


def build_graph(mask: np.ndarray, config: TopologyConfig) -> SkeletonGraph:
    result = skeletonize_mask(mask, connectivity=config.graph.connectivity)
    return SkeletonGraphBuilder(config.graph).build(result, image_id="synthetic", panel_id="synthetic")
