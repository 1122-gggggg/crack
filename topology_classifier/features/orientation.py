"""Orientation statistics for undirected line segments.

A skeleton path has no head or tail, so orientation is defined modulo 180 deg.
All circular statistics therefore use the doubled-angle representation
``(cos 2theta, sin 2theta)``, which makes ``theta`` and ``theta + 180`` identical.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np


@dataclass(frozen=True)
class OrientationStats:
    """Circular statistics of the segment orientations along a path."""

    dominant_deg: float
    anisotropy: float
    circular_variance: float
    entropy: float
    histogram: np.ndarray

    def as_dict(self, prefix: str = "orientation") -> Dict[str, float]:
        payload = {
            f"{prefix}_dominant_deg": self.dominant_deg,
            f"{prefix}_anisotropy": self.anisotropy,
            f"{prefix}_circular_variance": self.circular_variance,
            f"{prefix}_entropy": self.entropy,
        }
        for index, value in enumerate(self.histogram):
            payload[f"{prefix}_hist_{index:02d}"] = float(value)
        return payload


def _empty(bins: int) -> OrientationStats:
    return OrientationStats(
        dominant_deg=float("nan"),
        anisotropy=float("nan"),
        circular_variance=float("nan"),
        entropy=float("nan"),
        histogram=np.zeros(bins, dtype=np.float64),
    )


def segment_angles(path: np.ndarray, step: int = 1) -> np.ndarray:
    """Orientation in radians of each polyline segment, wrapped to ``[0, pi)``."""
    if path.shape[0] < step + 1:
        return np.empty(0, dtype=np.float64)
    deltas = path[step:].astype(np.float64) - path[:-step].astype(np.float64)
    keep = np.hypot(deltas[:, 0], deltas[:, 1]) > 1e-9
    deltas = deltas[keep]
    if deltas.size == 0:
        return np.empty(0, dtype=np.float64)
    angles = np.arctan2(deltas[:, 0], deltas[:, 1])  # (row, col) -> (y, x)
    return np.mod(angles, np.pi)


def segment_lengths(path: np.ndarray, step: int = 1) -> np.ndarray:
    if path.shape[0] < step + 1:
        return np.empty(0, dtype=np.float64)
    deltas = path[step:].astype(np.float64) - path[:-step].astype(np.float64)
    lengths = np.hypot(deltas[:, 0], deltas[:, 1])
    return lengths[lengths > 1e-9]


def orientation_stats(
    angles: np.ndarray,
    bins: int,
    weights: Optional[np.ndarray] = None,
) -> OrientationStats:
    """Circular statistics over orientations given in radians in ``[0, pi)``.

    Args:
        angles: Orientation of each segment, radians.
        bins: Number of histogram bins spanning ``[0, pi)``.
        weights: Optional per-segment weight (usually the segment length).

    Returns:
        Dominant orientation in degrees, anisotropy (``0`` isotropic, ``1``
        perfectly aligned), circular variance and normalised Shannon entropy of
        the orientation histogram.
    """
    if angles.size == 0:
        return _empty(bins)
    if weights is None:
        weights = np.ones_like(angles)
    weights = np.asarray(weights, dtype=np.float64)
    if weights.shape != angles.shape:
        raise ValueError(f"weights shape {weights.shape} != angles shape {angles.shape}")
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return _empty(bins)

    doubled = 2.0 * angles
    mean_cos = float(np.sum(weights * np.cos(doubled)) / total_weight)
    mean_sin = float(np.sum(weights * np.sin(doubled)) / total_weight)
    resultant = float(np.hypot(mean_cos, mean_sin))
    dominant = float(np.degrees(0.5 * np.arctan2(mean_sin, mean_cos)) % 180.0)

    histogram, _ = np.histogram(angles, bins=bins, range=(0.0, np.pi), weights=weights)
    histogram = histogram / max(float(histogram.sum()), 1e-12)
    nonzero = histogram[histogram > 0]
    entropy = float(-np.sum(nonzero * np.log(nonzero)) / np.log(bins)) if bins > 1 else 0.0

    return OrientationStats(
        dominant_deg=dominant,
        anisotropy=resultant,
        circular_variance=1.0 - resultant,
        entropy=entropy,
        histogram=histogram,
    )


def path_orientation_stats(path: np.ndarray, bins: int, step: int = 1) -> OrientationStats:
    """Length-weighted orientation statistics for one skeleton path."""
    angles = segment_angles(path, step=step)
    lengths = segment_lengths(path, step=step)
    if angles.size != lengths.size:  # defensive: both filters must agree
        size = min(angles.size, lengths.size)
        angles, lengths = angles[:size], lengths[:size]
    return orientation_stats(angles, bins=bins, weights=lengths)


def angular_difference_deg(a: float, b: float) -> float:
    """Smallest difference between two undirected orientations, in degrees."""
    diff = abs(a - b) % 180.0
    return float(min(diff, 180.0 - diff))
