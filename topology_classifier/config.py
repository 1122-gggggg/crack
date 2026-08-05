"""Typed configuration loaded from YAML.

Every tunable threshold of the pipeline lives here so that no magic number is
scattered across modules. The config also exposes a stable hash used as a cache
key: changing any value invalidates previously cached graphs and features.
"""
from __future__ import annotations

import hashlib
import json
import logging
from dataclasses import asdict, dataclass, field, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Type, TypeVar

import yaml

logger = logging.getLogger(__name__)

T = TypeVar("T")


@dataclass(frozen=True)
class DataConfig:
    images_dir: str = "data/images"
    rift_prob_dir: str = "data/rift_prob"
    rift_mask_dir: str = "data/rift_mask"
    labels_dir: str = "data/labels"
    edge_annotations_csv: Optional[str] = None
    panel_mapping_csv: Optional[str] = None
    output_dir: str = "outputs/topology"
    image_glob: str = "*.png,*.jpg,*.jpeg,*.tif,*.tiff"


@dataclass(frozen=True)
class ClassesConfig:
    background: int = 0
    crack: int = 1
    craquelure: int = 2
    other_line: int = 3
    ignore: int = 255
    enable_other_line: bool = True

    def active_names(self) -> list[str]:
        names = ["crack", "craquelure"]
        if self.enable_other_line:
            names.append("other_line")
        return names

    def pixel_value(self, name: str) -> int:
        return int(getattr(self, name))

    def name_to_index(self) -> Dict[str, int]:
        return {name: idx for idx, name in enumerate(self.active_names())}


@dataclass(frozen=True)
class PreprocessingConfig:
    high_threshold: float = 0.60
    low_threshold: float = 0.25
    binary_mask_threshold: int = 127
    minimum_skeleton_length: int = 10
    minimum_component_confidence: float = 0.0
    minimum_bbox_diagonal: float = 0.0
    minimum_elongation: float = 0.0
    enable_gap_repair: bool = False
    max_gap_pixels: int = 4
    max_angle_difference_deg: float = 25.0
    minimum_gap_confidence: float = 0.20
    gap_tangent_window: int = 7
    save_debug_masks: bool = True


@dataclass(frozen=True)
class GraphConfig:
    connectivity: int = 8
    junction_merge_radius: int = 2
    border_margin: int = 5
    orientation_bins: int = 12
    local_graph_hops: int = 2
    local_radius_pixels: int = 256
    minimum_enclosed_area: int = 16
    tolerant_face_closing_radius: int = 1
    curvature_resample_step: int = 5
    width_sample_stride: int = 1


@dataclass(frozen=True)
class AppearanceConfig:
    enable: bool = True
    side_offset_extra_px: float = 2.0
    patch_radius_px: int = 7
    max_samples_per_edge: int = 128
    enable_rift_encoder_features: bool = False


@dataclass(frozen=True)
class LabelsConfig:
    edge_label_purity: float = 0.80
    junction_ignore_radius: int = 3
    label_sample_uses_width: bool = True


@dataclass(frozen=True)
class ModelConfig:
    baseline_type: str = "xgboost"
    minimum_prediction_confidence: float = 0.60
    random_seed: int = 42
    junction_conflict_policy: str = "uncertain"


@dataclass(frozen=True)
class TrainingConfig:
    n_splits: int = 5
    group_column: str = "panel_id"
    class_weighting: str = "balanced"
    test_size: float = 0.25
    max_depth: int = 6
    n_estimators: int = 400
    learning_rate: float = 0.08


@dataclass(frozen=True)
class GNNConfig:
    hidden_dim: int = 96
    num_layers: int = 4
    dropout: float = 0.2
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    epochs: int = 100
    use_appearance_features: bool = True
    aux_graph_loss_weight: float = 0.0
    focal_gamma: float = 0.0


@dataclass(frozen=True)
class RuntimeConfig:
    cache_dir: str = "outputs/topology/cache"
    enable_cache: bool = True
    log_level: str = "INFO"
    errors_filename: str = "errors.jsonl"


@dataclass(frozen=True)
class TopologyConfig:
    data: DataConfig = field(default_factory=DataConfig)
    classes: ClassesConfig = field(default_factory=ClassesConfig)
    preprocessing: PreprocessingConfig = field(default_factory=PreprocessingConfig)
    graph: GraphConfig = field(default_factory=GraphConfig)
    appearance: AppearanceConfig = field(default_factory=AppearanceConfig)
    labels: LabelsConfig = field(default_factory=LabelsConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    gnn: GNNConfig = field(default_factory=GNNConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    source_path: Optional[str] = None

    @property
    def output_dir(self) -> Path:
        return Path(self.data.output_dir)

    @property
    def cache_dir(self) -> Path:
        return Path(self.runtime.cache_dir)

    def to_dict(self) -> Dict[str, Any]:
        payload = asdict(self)
        payload.pop("source_path", None)
        return payload

    def config_hash(self, *, scope: str = "all") -> str:
        """Stable hash of the config, optionally restricted to a scope subset."""
        payload = self.to_dict()
        if scope == "graph":
            payload = {k: payload[k] for k in ("preprocessing", "graph", "classes")}
        elif scope == "features":
            payload = {k: payload[k] for k in ("preprocessing", "graph", "classes", "appearance", "labels")}
        blob = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()[:16]


def _build(cls: Type[T], payload: Mapping[str, Any]) -> T:
    known = {f.name for f in fields(cls)}  # type: ignore[arg-type]
    unknown = set(payload) - known
    if unknown:
        logger.warning("ignoring unknown config keys for %s: %s", cls.__name__, sorted(unknown))
    return cls(**{k: v for k, v in payload.items() if k in known})  # type: ignore[call-arg]


def load_config(path: Optional[Path | str] = None, overrides: Optional[Mapping[str, Any]] = None) -> TopologyConfig:
    """Load a :class:`TopologyConfig` from YAML, falling back to defaults."""
    raw: Dict[str, Any] = {}
    source: Optional[str] = None
    if path is not None:
        cfg_path = Path(path)
        if not cfg_path.is_file():
            raise FileNotFoundError(f"Config not found: {cfg_path}")
        with cfg_path.open("r", encoding="utf-8") as handle:
            raw = yaml.safe_load(handle) or {}
        source = str(cfg_path.resolve())

    if overrides:
        for section, values in overrides.items():
            if isinstance(values, Mapping):
                raw.setdefault(section, {})
                raw[section].update(values)
            else:
                raw[section] = values

    sections: Dict[str, Any] = {}
    for f in fields(TopologyConfig):
        if f.name == "source_path":
            continue
        section_cls = f.type if is_dataclass(f.type) else None
        section_payload = raw.get(f.name, {}) or {}
        if section_cls is None:
            section_cls = {
                "data": DataConfig,
                "classes": ClassesConfig,
                "preprocessing": PreprocessingConfig,
                "graph": GraphConfig,
                "appearance": AppearanceConfig,
                "labels": LabelsConfig,
                "model": ModelConfig,
                "training": TrainingConfig,
                "gnn": GNNConfig,
                "runtime": RuntimeConfig,
            }[f.name]
        sections[f.name] = _build(section_cls, section_payload)

    return TopologyConfig(source_path=source, **sections)


def dump_config(config: TopologyConfig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(config.to_dict(), handle, sort_keys=False, allow_unicode=True)
