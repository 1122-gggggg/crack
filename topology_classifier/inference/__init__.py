"""Prediction and pixel-space rasterization."""
from .classify_edges import ImagePrediction, classify_image, infer_dataset
from .rasterize import RasterizationResult, class_mask_to_color, nearest_edge_map, rasterize_predictions

__all__ = [
    "ImagePrediction",
    "RasterizationResult",
    "class_mask_to_color",
    "classify_image",
    "infer_dataset",
    "nearest_edge_map",
    "rasterize_predictions",
]
