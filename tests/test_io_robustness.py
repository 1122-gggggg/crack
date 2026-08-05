"""Input discovery and validation should fail clearly on ambiguous data."""
from __future__ import annotations

import numpy as np
import pytest

from topology_classifier.io.dataset_adapter import _strip_inference_tag
from topology_classifier.io.rift_adapter import RiftAdapter
from topology_classifier.preprocessing.hysteresis import hysteresis_threshold


def test_inference_tags_do_not_confuse_prefix_image_ids(tmp_path):
    np.save(tmp_path / "foo_prob.npy", np.ones((2, 2), dtype=np.float32))
    np.save(tmp_path / "foobar_prob.npy", np.ones((2, 2), dtype=np.float32))
    adapter = RiftAdapter(tmp_path)

    assert adapter.find_probability("foo") == tmp_path / "foo_prob.npy"
    assert _strip_inference_tag("foo_s1_model_prob", ["foo", "foobar"]) == "foo"
    assert _strip_inference_tag("foobar_s1_model_prob", ["foo", "foobar"]) == "foobar"


def test_empty_or_nonfinite_probability_is_rejected(tmp_path):
    np.save(tmp_path / "empty_prob.npy", np.empty((0, 0), dtype=np.float32))
    with pytest.raises(ValueError, match="empty"):
        RiftAdapter(tmp_path).load("empty")

    with pytest.raises(ValueError, match="thresholds"):
        hysteresis_threshold(np.ones((2, 2), dtype=np.float32), high=1.2, low=0.2)
