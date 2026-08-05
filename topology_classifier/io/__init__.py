"""Input adapters for Stage-1 (RIFT) artefacts and dataset metadata."""
from .dataset_adapter import DatasetAdapter, ImageRecord
from .rift_adapter import MissingRiftOutputError, RiftAdapter, RiftOutputs

__all__ = [
    "DatasetAdapter",
    "ImageRecord",
    "MissingRiftOutputError",
    "RiftAdapter",
    "RiftOutputs",
]
