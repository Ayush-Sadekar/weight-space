#!/usr/bin/env python3
"""Edit a single weight operation on task/model.npz. Run from a run's task/ directory."""

from __future__ import annotations

import _pathfix  # noqa: F401  — before numpy (avoid shadowing stdlib inspect)

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from _common import (
    ensure_under_task,
    get_layer,
    load_model,
    parse_slice,
    save_model,
    task_dir,
)


def append_edit(op: str, args: dict, n_changed: int) -> None:
    path = task_dir() / "edits.jsonl"
    rec = {
        "timestamp": time.time(),
        "op": op,
        "args": args,
        "n_changed": int(n_changed),
    }
    with path.open("a", encoding="utf-8") as f:
        f.write(json.dumps(rec, sort_keys=True) + "\n")


def require_2d(arr: np.ndarray, op: str) -> None:
    if arr.ndim != 2:
        raise SystemExit(f"error: {op} requires a 2D layer, got shape {list(arr.shape)}")


def cmd_set(args: argparse.Namespace) -> int:
    model = load_model()
    arr = get_layer(model, args.layer)
    require_2d(arr, "set")
    i, j = args.i, args.j
    if i < 0 or i >= arr.shape[0] or j < 0 or j >= arr.shape[1]:
        raise SystemExit(f"error: index ({i},{j}) out of range for shape {list(arr.shape)}")
    arr[i, j] = float(args.value)
    save_model(model)
    append_edit("set", {"layer": args.layer, "i": i, "j": j, "value": float(args.value)}, 1)
    print(f"set {args.layer}[{i},{j}]={args.value}  changed=1")
    return 0


def cmd_add(args: argparse.Namespace) -> int:
    model = load_model()
    arr = get_layer(model, args.layer)
    require_2d(arr, "add")
    i, j = args.i, args.j
    if i < 0 or i >= arr.shape[0] or j < 0 or j >= arr.shape[1]:
        raise SystemExit(f"error: index ({i},{j}) out of range for shape {list(arr.shape)}")
    arr[i, j] = arr[i, j] + float(args.delta)
    save_model(model)
    append_edit("add", {"layer": args.layer, "i": i, "j": j, "delta": float(args.delta)}, 1)
    print(f"add {args.layer}[{i},{j}]+={args.delta}  changed=1")
    return 0


def cmd_scale(args: argparse.Namespace) -> int:
    model = load_model()
    arr = get_layer(model, args.layer)
    if arr.ndim == 1:
        rs = parse_slice(args.rows, arr.shape[0])
        view = arr[rs]
        n = int(view.size)
        arr[rs] = view * float(args.factor)
        save_model(model)
        append_edit(
            "scale",
            {"layer": args.layer, "rows": args.rows, "cols": args.cols, "factor": float(args.factor)},
            n,
        )
        print(f"scale {args.layer} factor={args.factor}  changed={n}")
        return 0
    require_2d(arr, "scale")
    rs = parse_slice(args.rows, arr.shape[0])
    cs = parse_slice(args.cols, arr.shape[1])
    view = arr[rs, cs]
    n = int(view.size)
    arr[rs, cs] = view * float(args.factor)
    save_model(model)
    append_edit(
        "scale",
        {"layer": args.layer, "rows": args.rows, "cols": args.cols, "factor": float(args.factor)},
        n,
    )
    print(f"scale {args.layer} rows={args.rows} cols={args.cols} factor={args.factor}  changed={n}")
    return 0


def cmd_zero(args: argparse.Namespace) -> int:
    model = load_model()
    arr = get_layer(model, args.layer)
    if arr.ndim == 1:
        rs = parse_slice(args.rows, arr.shape[0])
        n = int(arr[rs].size)
        arr[rs] = 0.0
        save_model(model)
        append_edit("zero", {"layer": args.layer, "rows": args.rows, "cols": args.cols}, n)
        print(f"zero {args.layer}  changed={n}")
        return 0
    require_2d(arr, "zero")
    rs = parse_slice(args.rows, arr.shape[0])
    cs = parse_slice(args.cols, arr.shape[1])
    n = int(arr[rs, cs].size)
    arr[rs, cs] = 0.0
    save_model(model)
    append_edit("zero", {"layer": args.layer, "rows": args.rows, "cols": args.cols}, n)
    print(f"zero {args.layer} rows={args.rows} cols={args.cols}  changed={n}")
    return 0


def cmd_ablate(args: argparse.Namespace) -> int:
    model = load_model()
    arr = get_layer(model, args.layer)
    require_2d(arr, "ablate")
    # Only allow vectors under task/ (typically task/vectors/)
    resolved = ensure_under_task(Path(args.vector))
    if not resolved.is_file():
        raise SystemExit(f"error: vector file not found: {args.vector}")

    vec = np.load(resolved).astype(np.float64).ravel()
    side = args.side
    if side == "out":
        # project direction out of rows: W <- (I - vv^T/||v||^2) W
        if vec.shape[0] != arr.shape[0]:
            raise SystemExit(
                f"error: out-side vector length {vec.shape[0]} != rows {arr.shape[0]}"
            )
        nrm2 = float(vec @ vec)
        if nrm2 < 1e-18:
            raise SystemExit("error: zero vector")
        proj = np.outer(vec, vec) / nrm2
        before = arr.copy()
        arr[:] = arr - proj @ arr
    else:  # in
        if vec.shape[0] != arr.shape[1]:
            raise SystemExit(
                f"error: in-side vector length {vec.shape[0]} != cols {arr.shape[1]}"
            )
        nrm2 = float(vec @ vec)
        if nrm2 < 1e-18:
            raise SystemExit("error: zero vector")
        before = arr.copy()
        # W <- W (I - vv^T/||v||^2)
        proj = np.outer(vec, vec) / nrm2
        arr[:] = arr - arr @ proj

    n_changed = int(np.sum(before != arr))
    save_model(model)
    append_edit(
        "ablate",
        {
            "layer": args.layer,
            "vector": str(resolved.relative_to(task_dir())),
            "side": side,
        },
        n_changed,
    )
    print(f"ablate {args.layer} side={side} vector={resolved.name}  changed={n_changed}")
    return 0


def cmd_save_vector(args: argparse.Namespace) -> int:
    name = args.name
    if "/" in name or "\\" in name or name.startswith("."):
        raise SystemExit("error: vector name must be a plain filename stem (no path)")
    try:
        values = [float(x.strip()) for x in args.values.split(",") if x.strip() != ""]
    except ValueError:
        raise SystemExit("error: values must be comma-separated floats")
    if not values:
        raise SystemExit("error: empty vector")
    vec_dir = task_dir() / "vectors"
    vec_dir.mkdir(exist_ok=True)
    out = ensure_under_task(Path("vectors") / f"{name}.npy")
    np.save(out, np.asarray(values, dtype=np.float64))
    append_edit("save-vector", {"name": name, "dim": len(values)}, len(values))
    print(f"saved vectors/{name}.npy  dim={len(values)}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="edit.py", description="Edit model.npz (one op per call)")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("set")
    s.add_argument("layer")
    s.add_argument("i", type=int)
    s.add_argument("j", type=int)
    s.add_argument("value", type=float)

    a = sub.add_parser("add")
    a.add_argument("layer")
    a.add_argument("i", type=int)
    a.add_argument("j", type=int)
    a.add_argument("delta", type=float)

    sc = sub.add_parser("scale")
    sc.add_argument("layer")
    sc.add_argument("--rows", default=None)
    sc.add_argument("--cols", default=None)
    sc.add_argument("factor", type=float)

    z = sub.add_parser("zero")
    z.add_argument("layer")
    z.add_argument("--rows", default=None)
    z.add_argument("--cols", default=None)

    ab = sub.add_parser("ablate")
    ab.add_argument("layer")
    ab.add_argument("vector", help="path to .npy under task/ (e.g. vectors/v.npy)")
    ab.add_argument("--side", choices=("out", "in"), default="out")

    sv = sub.add_parser("save-vector")
    sv.add_argument("name")
    sv.add_argument("values", help="comma-separated floats")

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "set": cmd_set,
        "add": cmd_add,
        "scale": cmd_scale,
        "zero": cmd_zero,
        "ablate": cmd_ablate,
        "save-vector": cmd_save_vector,
    }
    try:
        return handlers[args.cmd](args)
    except SystemExit:
        raise
    except Exception as e:  # pragma: no cover
        print(f"error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
