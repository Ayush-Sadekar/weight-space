"""Shared helpers for inspect/edit/probe. Not an allowed Claude command."""

from __future__ import annotations

import os
import sys
from pathlib import Path

import numpy as np

# edit-weights root
ROOT = Path(__file__).resolve().parents[1]
HARNESS = ROOT / "harness"
if str(HARNESS) not in sys.path:
    sys.path.insert(0, str(HARNESS))

from models import MLP  # noqa: E402


def task_dir() -> Path:
    """Working directory must be the run's task/ folder."""
    return Path.cwd().resolve()


def model_path() -> Path:
    p = task_dir() / "model.npz"
    if not p.is_file():
        raise SystemExit(f"error: model.npz not found in {task_dir()}")
    return p


def load_model() -> MLP:
    return MLP.load(model_path())


def save_model(model: MLP) -> None:
    model.save(model_path())


def ensure_under_task(path: Path) -> Path:
    """Resolve path and refuse anything outside task/."""
    task = task_dir()
    resolved = (task / path).resolve() if not path.is_absolute() else path.resolve()
    try:
        resolved.relative_to(task)
    except ValueError:
        raise SystemExit(f"error: path escapes task/: {path}")
    return resolved


def parse_slice(spec: str | None, size: int) -> slice:
    """Parse 'a:b' / ':b' / 'a:' / 'a' into a Python slice clamped to [0, size)."""
    if spec is None:
        return slice(0, size)
    spec = spec.strip()
    if ":" not in spec:
        i = int(spec)
        if i < 0 or i >= size:
            raise SystemExit(f"error: index {i} out of range for size {size}")
        return slice(i, i + 1)
    parts = spec.split(":", 1)
    start = 0 if parts[0] == "" else int(parts[0])
    stop = size if parts[1] == "" else int(parts[1])
    if start < 0 or stop < 0 or start > size or stop > size or start > stop:
        raise SystemExit(f"error: bad slice {spec!r} for size {size}")
    return slice(start, stop)


def get_layer(model: MLP, name: str) -> np.ndarray:
    if name not in model.weights:
        known = ", ".join(model.layer_names())
        raise SystemExit(f"error: unknown layer {name!r}; known: {known}")
    return model.weights[name]


def env_precision(default: int = 4) -> int:
    raw = os.environ.get("WS_PRECISION", str(default))
    try:
        p = int(raw)
    except ValueError:
        raise SystemExit(f"error: WS_PRECISION must be int, got {raw!r}")
    if p < 0 or p > 16:
        raise SystemExit(f"error: WS_PRECISION out of range: {p}")
    return p


def env_max_print(default: int = 4096) -> int:
    raw = os.environ.get("WS_MAX_PRINT", str(default))
    try:
        n = int(raw)
    except ValueError:
        raise SystemExit(f"error: WS_MAX_PRINT must be int, got {raw!r}")
    if n < 1:
        raise SystemExit(f"error: WS_MAX_PRINT must be >= 1")
    return n


def fmt_float(x: float, precision: int | None = None) -> str:
    p = env_precision() if precision is None else precision
    return f"{x:.{p}g}"
