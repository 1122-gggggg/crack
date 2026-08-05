"""Logging helpers and the per-image failure journal."""
from __future__ import annotations

import json
import logging
import platform
import subprocess
import sys
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


def setup_logging(level: str = "INFO") -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s",
        stream=sys.stdout,
        force=True,
    )


class ErrorJournal:
    """Append-only ``errors.jsonl`` writer so no failing image is silently skipped."""

    def __init__(self, path: Path) -> None:
        self.path = path
        self.count = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, image_id: str, stage: str, error: BaseException, extra: Optional[Dict[str, Any]] = None) -> None:
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "image_id": image_id,
            "stage": stage,
            "error_type": type(error).__name__,
            "message": str(error),
            "traceback": "".join(traceback.format_exception(type(error), error, error.__traceback__))[-4000:],
        }
        if extra:
            entry.update(extra)
        with self.path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(entry, ensure_ascii=False) + "\n")
        self.count += 1
        logger.error("[%s] %s failed: %s: %s", stage, image_id, type(error).__name__, error)


def environment_report() -> Dict[str, Any]:
    """Capture interpreter, platform and dependency versions for reproducibility."""
    packages: Dict[str, str] = {}
    for name in (
        "numpy",
        "scipy",
        "cv2",
        "skimage",
        "networkx",
        "pandas",
        "sklearn",
        "xgboost",
        "torch",
        "torch_geometric",
    ):
        try:
            module = __import__(name)
        except ImportError:
            packages[name] = "not installed"
            continue
        packages[name] = str(getattr(module, "__version__", "unknown"))

    report: Dict[str, Any] = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "python": sys.version,
        "platform": platform.platform(),
        "executable": sys.executable,
        "packages": packages,
    }
    try:
        rev = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            capture_output=True,
            text=True,
            timeout=10,
            check=False,
            cwd=str(Path(__file__).resolve().parent.parent),
        )
        report["git_commit"] = rev.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        report["git_commit"] = "unknown"
    return report


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, default=_json_default)


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    return str(value)
