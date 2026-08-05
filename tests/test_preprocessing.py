"""Thresholding, cleanup and gap repair behaviour."""
from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

from conftest import draw_line
from topology_classifier.preprocessing import (
    binarize,
    hysteresis_threshold,
    preprocess_mask,
    remove_noise_components,
    repair_gaps,
)


def _probability_from_mask(mask: np.ndarray, high: float = 0.9, low: float = 0.35) -> np.ndarray:
    prob = np.zeros(mask.shape, dtype=np.float32)
    prob[mask.astype(bool)] = high
    return prob


def test_hysteresis_keeps_connected_weak_pixels():
    prob = np.zeros((20, 20), dtype=np.float32)
    prob[10, 2:10] = 0.9
    prob[10, 10:16] = 0.4  # weak but attached to the strong run
    prob[3, 2:10] = 0.4  # weak and isolated
    mask = hysteresis_threshold(prob, high=0.6, low=0.25)
    assert mask[10, 2:16].all()
    assert not mask[3].any()


def test_hysteresis_rejects_inverted_thresholds():
    with pytest.raises(ValueError):
        hysteresis_threshold(np.zeros((4, 4), dtype=np.float32), high=0.2, low=0.8)


def test_binarize_falls_back_to_mask(config, caplog):
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[8, 2:12] = 255
    with caplog.at_level("WARNING"):
        result = binarize(None, mask, config.preprocessing)
    assert not result.probability_available
    assert result.pixel_count == 10
    assert "probability feature unavailable" in caplog.text


def test_binarize_requires_some_input(config):
    with pytest.raises(ValueError):
        binarize(None, None, config.preprocessing)


def test_cleanup_removes_short_specks_but_keeps_thin_lines(config):
    mask = np.zeros((120, 120), dtype=np.uint8)
    draw_line(mask, (60, 10), (60, 110), thickness=1)  # thin, long, low area
    mask[10, 10] = 1  # speck
    mask[20, 20:23] = 1  # short stub
    result = remove_noise_components(mask, config.preprocessing)
    assert result.kept_component_count == 1
    assert result.removed_component_count == 2
    assert result.mask[60, 10:110].all()


def test_gap_repair_disabled_by_default(config):
    mask = np.zeros((40, 40), dtype=np.uint8)
    draw_line(mask, (20, 5), (20, 18), thickness=1)
    draw_line(mask, (20, 22), (20, 35), thickness=1)
    result = repair_gaps(mask, config.preprocessing)
    assert not result.enabled
    assert not result.bridges
    assert not result.mask[20, 19:22].any()


def test_gap_repair_bridges_aligned_endpoints(config):
    enabled = replace(config.preprocessing, enable_gap_repair=True, max_gap_pixels=6)
    mask = np.zeros((40, 40), dtype=np.uint8)
    draw_line(mask, (20, 5), (20, 18), thickness=1)
    draw_line(mask, (20, 22), (20, 35), thickness=1)
    probability = _probability_from_mask(mask)
    probability[20, 19:22] = 0.4  # faint but above minimum_gap_confidence
    result = repair_gaps(mask, enabled, probability=probability)
    assert result.enabled
    assert len(result.bridges) == 1
    assert result.mask[20, 19:22].all()


def test_gap_repair_rejects_perpendicular_endpoints(config):
    enabled = replace(config.preprocessing, enable_gap_repair=True, max_gap_pixels=6)
    mask = np.zeros((40, 40), dtype=np.uint8)
    draw_line(mask, (20, 5), (20, 18), thickness=1)
    draw_line(mask, (21, 22), (34, 22), thickness=1)  # runs away at 90 degrees
    probability = _probability_from_mask(mask)
    probability[20:22, 19:23] = 0.5
    result = repair_gaps(mask, enabled, probability=probability)
    assert not result.bridges
    assert result.rejected.get("angle", 0) >= 1


def test_gap_repair_rejects_unsupported_gap(config):
    enabled = replace(config.preprocessing, enable_gap_repair=True, max_gap_pixels=6)
    mask = np.zeros((40, 40), dtype=np.uint8)
    draw_line(mask, (20, 5), (20, 18), thickness=1)
    draw_line(mask, (20, 22), (20, 35), thickness=1)
    probability = _probability_from_mask(mask)  # gap pixels stay at 0.0
    result = repair_gaps(mask, enabled, probability=probability)
    assert not result.bridges
    assert result.rejected.get("confidence", 0) >= 1


def test_preprocess_mask_reports_every_stage(config):
    mask = np.zeros((120, 120), dtype=np.uint8)
    draw_line(mask, (60, 10), (60, 110), thickness=2)
    mask[10, 10] = 1
    probability = _probability_from_mask(mask)
    result = preprocess_mask(config.preprocessing, probability=probability, image_id="unit")
    payload = result.as_dict()
    assert payload["threshold"]["probability_available"] is True
    assert payload["cleanup"]["removed_component_count"] == 1
    assert payload["gap_repair"]["enabled"] is False
    assert payload["final_pixel_count"] > 100
