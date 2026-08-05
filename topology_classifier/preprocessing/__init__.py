"""Topology-preserving preprocessing of RIFT outputs."""
from .cleanup import CleanupResult, component_statistics, remove_noise_components
from .gap_repair import GapBridge, GapRepairResult, gap_repair_debug_image, repair_gaps
from .hysteresis import ThresholdResult, binarize, hysteresis_threshold
from .pipeline import PreprocessedMask, preprocess_mask

__all__ = [
    "CleanupResult",
    "GapBridge",
    "GapRepairResult",
    "PreprocessedMask",
    "ThresholdResult",
    "binarize",
    "component_statistics",
    "gap_repair_debug_image",
    "hysteresis_threshold",
    "preprocess_mask",
    "remove_noise_components",
    "repair_gaps",
]
