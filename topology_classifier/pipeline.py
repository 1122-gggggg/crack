"""Per-image orchestration: RIFT output -> preprocessed mask -> skeleton graph.

Caching is keyed by the graph-scope config hash, so any change to a
preprocessing or graph threshold produces a different cache directory and old
artefacts can never be reused by mistake.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from .config import TopologyConfig
from .graph.graph_builder import SkeletonGraphBuilder
from .graph.graph_types import SkeletonGraph
from .graph.graph_validation import validate_graph
from .graph.skeletonize import skeletonize_mask
from .io.dataset_adapter import DatasetAdapter, ImageRecord
from .logging_utils import ErrorJournal
from .preprocessing.pipeline import preprocess_mask

logger = logging.getLogger(__name__)


@dataclass
class ImageArtifacts:
    """Everything downstream stages need for one image."""

    record: ImageRecord
    graph: SkeletonGraph
    mask: np.ndarray
    probability: Optional[np.ndarray]
    probability_available: bool
    stats: Dict[str, object] = field(default_factory=dict)
    from_cache: bool = False
    _distance: Optional[np.ndarray] = None

    @property
    def image_id(self) -> str:
        return self.record.image_id

    def distance(self) -> np.ndarray:
        """Euclidean distance transform of the mask (line half-width)."""
        if self._distance is None:
            import cv2

            self._distance = cv2.distanceTransform(self.mask.astype(np.uint8), cv2.DIST_L2, 5)
        return self._distance


def _mask_cache_path(directory: Path, image_id: str) -> Path:
    return directory / f"{image_id}_mask.npz"


def _graph_cache_path(directory: Path, image_id: str) -> Path:
    return directory / f"{image_id}_graph.npz"


def _stats_cache_path(directory: Path, image_id: str) -> Path:
    return directory / f"{image_id}_stats.json"


def _input_signature(rift) -> Dict[str, object]:
    """Capture the Stage-1 files used to build a cache entry.

    A config hash alone cannot invalidate a graph when a new RIFT prediction is
    written under the same image id.  File size and nanosecond mtime avoid
    hashing 160 MB probability maps on every invocation while still detecting
    normal replacement/rewrites.
    """

    signature: Dict[str, object] = {}
    for name, path in (("probability", rift.probability_path), ("mask", rift.mask_path)):
        if path is None:
            signature[name] = None
            continue
        resolved = Path(path).resolve()
        try:
            stat = resolved.stat()
        except OSError:
            signature[name] = {"path": str(resolved), "missing": True}
            continue
        signature[name] = {
            "path": str(resolved),
            "size": int(stat.st_size),
            "mtime_ns": int(stat.st_mtime_ns),
        }
    return signature


def cache_directory(config: TopologyConfig) -> Path:
    return config.cache_dir / "graphs" / config.config_hash(scope="graph")


def _save_mask(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        packed=np.packbits(mask.astype(bool), axis=None),
        shape=np.asarray(mask.shape, dtype=np.int64),
    )


def _load_mask(path: Path) -> np.ndarray:
    with np.load(path) as data:
        shape = tuple(int(v) for v in data["shape"])
        count = int(np.prod(shape))
        return np.unpackbits(data["packed"])[:count].astype(bool).reshape(shape)


def build_image_graph(
    config: TopologyConfig,
    record: ImageRecord,
    adapter: DatasetAdapter,
    use_cache: Optional[bool] = None,
    debug_dir: Optional[Path] = None,
) -> ImageArtifacts:
    """Load Stage-1 output, preprocess it and build the skeleton graph.

    Raises:
        MissingRiftOutputError: If the image has no Stage-1 output at all.
        ValueError: If the graph fails structural validation.
    """
    enabled = config.runtime.enable_cache if use_cache is None else use_cache
    directory = cache_directory(config)
    graph_path = _graph_cache_path(directory, record.image_id)
    mask_path = _mask_cache_path(directory, record.image_id)
    stats_path = _stats_cache_path(directory, record.image_id)

    rift = adapter.load_rift(record)

    input_signature = _input_signature(rift)
    if enabled and graph_path.is_file() and mask_path.is_file() and stats_path.is_file():
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        if stats.get("input_signature") == input_signature:
            graph = SkeletonGraph.load(graph_path)
            mask = _load_mask(mask_path)
            logger.info("%s: graph loaded from cache (%s)", record.image_id, directory.name)
            return ImageArtifacts(
                record=record,
                graph=graph,
                mask=mask,
                probability=rift.probability,
                probability_available=rift.probability_available,
                stats=stats,
                from_cache=True,
            )
        logger.info("%s: cache is stale; rebuilding graph", record.image_id)

    preprocessed = preprocess_mask(
        config.preprocessing,
        probability=rift.probability,
        mask=rift.mask,
        debug_dir=debug_dir,
        image_id=record.image_id,
    )
    skeleton = skeletonize_mask(preprocessed.mask, connectivity=config.graph.connectivity)
    graph = SkeletonGraphBuilder(config.graph).build(
        skeleton, image_id=record.image_id, panel_id=record.panel_id
    )
    report = validate_graph(graph)
    if not report.is_valid:
        raise ValueError(f"{record.image_id}: invalid skeleton graph: {report.errors[:5]}")
    if report.warnings:
        logger.warning("%s: %d graph warning(s): %s", record.image_id, len(report.warnings), report.warnings[:3])

    stats: Dict[str, object] = {
        "config_hash": config.config_hash(scope="graph"),
        "input_signature": input_signature,
        "probability_available": rift.probability_available,
        "probability_path": str(rift.probability_path) if rift.probability_path else None,
        "mask_path": str(rift.mask_path) if rift.mask_path else None,
        **preprocessed.as_dict(),
        **{k: float(v) for k, v in graph.summary().items()},
        "graph_warnings": report.warnings[:20],
    }

    if enabled:
        graph.save(graph_path)
        _save_mask(mask_path, preprocessed.mask)
        stats_path.parent.mkdir(parents=True, exist_ok=True)
        stats_path.write_text(json.dumps(stats, indent=2, default=str), encoding="utf-8")

    return ImageArtifacts(
        record=record,
        graph=graph,
        mask=preprocessed.mask,
        probability=rift.probability,
        probability_available=rift.probability_available,
        stats=stats,
    )


def build_all_graphs(
    config: TopologyConfig,
    adapter: Optional[DatasetAdapter] = None,
    journal: Optional[ErrorJournal] = None,
    use_cache: Optional[bool] = None,
    limit: Optional[int] = None,
) -> List[ImageArtifacts]:
    """Build graphs for every discovered image, recording failures rather than hiding them."""
    adapter = adapter or DatasetAdapter(config)
    journal = journal or ErrorJournal(config.output_dir / config.runtime.errors_filename)
    records = adapter.records()
    if limit is not None:
        records = records[:limit]

    artifacts: List[ImageArtifacts] = []
    debug_root = config.output_dir / "debug" if config.preprocessing.save_debug_masks else None
    for index, record in enumerate(records, start=1):
        logger.info("[%d/%d] %s", index, len(records), record.image_id)
        try:
            artifacts.append(
                build_image_graph(config, record, adapter, use_cache=use_cache, debug_dir=debug_root)
            )
        except (OSError, ValueError, KeyError, MemoryError) as error:
            journal.record(record.image_id, stage="build_graph", error=error)
            logger.error("%s: graph construction failed (%s)", record.image_id, error)

    logger.info("graphs built for %d/%d image(s)", len(artifacts), len(records))
    return artifacts
