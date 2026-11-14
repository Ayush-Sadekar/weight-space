#!/usr/bin/env python3
"""Forward-pass probe with a budget stored in ../hidden/probe_budget.json."""

from __future__ import annotations

import _pathfix  # noqa: F401  — before numpy (avoid shadowing stdlib inspect)

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from _common import env_precision, fmt_float, load_model, task_dir


def budget_path() -> Path:
    # task/ -> run_id/ -> hidden/
    return task_dir().parent / "hidden" / "probe_budget.json"


def read_and_decrement_budget() -> int | None:
    """Return remaining budget after decrement, or None if disabled.

    Raises SystemExit on exhaustion / missing file.
    """
    path = budget_path()
    if not path.is_file():
        raise SystemExit(f"error: probe budget file missing: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if data.get("disabled") is True or data.get("status") == "disabled":
        print("probes disabled for this condition")
        raise SystemExit(2)

    remaining = int(data.get("remaining", data.get("budget", 0)))
    if remaining <= 0:
        raise SystemExit("error: probe budget exhausted (0 remaining)")

    data["remaining"] = remaining - 1
    data["used"] = int(data.get("used", 0)) + 1
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")
    return data["remaining"]


def parse_input(args: argparse.Namespace) -> np.ndarray:
    if args.inline is not None:
        try:
            raw = json.loads(args.inline)
        except json.JSONDecodeError as e:
            raise SystemExit(f"error: bad --inline JSON: {e}")
    else:
        path = Path(args.input_json)
        # Refuse reading outside task/ for input files
        task = task_dir()
        resolved = path if path.is_absolute() else (task / path)
        resolved = resolved.resolve()
        try:
            resolved.relative_to(task)
        except ValueError:
            raise SystemExit(f"error: input path escapes task/: {path}")
        if not resolved.is_file():
            raise SystemExit(f"error: input file not found: {path}")
        with resolved.open("r", encoding="utf-8") as f:
            raw = json.load(f)

    arr = np.asarray(raw, dtype=np.float64)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise SystemExit(f"error: input must be 1D or 2D, got shape {arr.shape}")
    return arr


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="probe.py", description="Forward pass with probe budget")
    parser.add_argument(
        "input_json",
        nargs="?",
        default=None,
        help="JSON file under task/ with a vector or batch",
    )
    parser.add_argument(
        "--inline",
        default=None,
        help='Inline JSON vector, e.g. --inline "[0.1, 0.2, ...]"',
    )
    args = parser.parse_args(argv)

    if args.input_json is None and args.inline is None:
        parser.error("provide input.json or --inline")
    if args.input_json is not None and args.inline is not None:
        parser.error("provide only one of input.json or --inline")

    # Validate input (and model shape) before spending budget.
    X = parse_input(args)
    model = load_model()
    expected = model.layer_sizes[0]
    if X.shape[1] != expected:
        raise SystemExit(f"error: input dim {X.shape[1]} != model input {expected}")

    remaining = read_and_decrement_budget()
    logits = model.forward(X)
    preds = logits.argmax(axis=-1)
    p = env_precision()

    print(f"batch={X.shape[0]}  input_dim={X.shape[1]}  probes_remaining={remaining}")
    for i in range(X.shape[0]):
        logit_str = " ".join(fmt_float(float(v), p) for v in logits[i])
        print(f"[{i}] logits=[{logit_str}]  argmax={int(preds[i])}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except SystemExit:
        raise
    except Exception as e:  # pragma: no cover
        print(f"error: {e}", file=sys.stderr)
        raise SystemExit(1)
