#!/usr/bin/env python3
"""Inspect weight matrices in task/model.npz. Run from a run's task/ directory."""

from __future__ import annotations

import _pathfix  # noqa: F401  — before numpy (avoid shadowing stdlib inspect)

import argparse
import sys

import numpy as np

from _common import (
    env_max_print,
    env_precision,
    fmt_float,
    get_layer,
    load_model,
    parse_slice,
)


def cmd_list(_: argparse.Namespace) -> int:
    model = load_model()
    print(f"layers: {len(model.layer_names())}  sizes: {model.layer_sizes}")
    for name in model.layer_names():
        arr = model.weights[name]
        print(f"{name}\tshape={list(arr.shape)}\tdtype={arr.dtype}")
    return 0


def cmd_stats(args: argparse.Namespace) -> int:
    model = load_model()
    arr = get_layer(model, args.layer)
    flat = arr.ravel()
    near_zero = float(np.mean(np.abs(flat) < 1e-6) * 100.0)
    p = env_precision()
    print(f"layer={args.layer} shape={list(arr.shape)} dtype={arr.dtype}")
    print(
        f"mean={fmt_float(float(flat.mean()), p)}  "
        f"std={fmt_float(float(flat.std()), p)}  "
        f"min={fmt_float(float(flat.min()), p)}  "
        f"max={fmt_float(float(flat.max()), p)}"
    )
    print(f"pct_near_zero(<1e-6)={near_zero:.2f}")
    if arr.ndim == 2:
        row_norms = np.linalg.norm(arr, axis=1)
        col_norms = np.linalg.norm(arr, axis=0)
        print(
            f"row_norms: mean={fmt_float(float(row_norms.mean()), p)}  "
            f"min={fmt_float(float(row_norms.min()), p)}  "
            f"max={fmt_float(float(row_norms.max()), p)}"
        )
        print(
            f"col_norms: mean={fmt_float(float(col_norms.mean()), p)}  "
            f"min={fmt_float(float(col_norms.min()), p)}  "
            f"max={fmt_float(float(col_norms.max()), p)}"
        )
        # compact per-row / per-col summary (indices of extremes)
        print(f"row_norm_argmax={int(row_norms.argmax())}  row_norm_argmin={int(row_norms.argmin())}")
        print(f"col_norm_argmax={int(col_norms.argmax())}  col_norm_argmin={int(col_norms.argmin())}")
    elif arr.ndim == 1:
        print(f"l2_norm={fmt_float(float(np.linalg.norm(arr)), p)}")
    return 0


def cmd_show(args: argparse.Namespace) -> int:
    model = load_model()
    arr = get_layer(model, args.layer)
    precision = args.precision if args.precision is not None else env_precision()
    max_print = env_max_print()

    if arr.ndim == 1:
        sl = parse_slice(args.rows, arr.shape[0])
        view = arr[sl]
        if view.size > max_print:
            raise SystemExit(
                f"error: refusing to print {view.size} values (MAX_PRINT={max_print}); "
                "narrow --rows"
            )
        print(f"layer={args.layer} slice=[{sl.start}:{sl.stop}] shape={list(view.shape)}")
        np.set_printoptions(precision=precision, suppress=True, linewidth=120, threshold=max_print)
        print(np.array2string(view, precision=precision, separator=", "))
        return 0

    if arr.ndim != 2:
        raise SystemExit(f"error: show supports 1D/2D layers, got ndim={arr.ndim}")

    rs = parse_slice(args.rows, arr.shape[0])
    cs = parse_slice(args.cols, arr.shape[1])
    view = arr[rs, cs]
    if view.size > max_print:
        raise SystemExit(
            f"error: refusing to print {view.size} values (MAX_PRINT={max_print}); "
            "narrow --rows/--cols"
        )
    print(
        f"layer={args.layer} rows=[{rs.start}:{rs.stop}] cols=[{cs.start}:{cs.stop}] "
        f"shape={list(view.shape)}"
    )
    np.set_printoptions(precision=precision, suppress=True, linewidth=120, threshold=max_print)
    print(np.array2string(view, precision=precision, separator=", "))
    return 0


def cmd_svd(args: argparse.Namespace) -> int:
    model = load_model()
    arr = get_layer(model, args.layer)
    if arr.ndim != 2:
        raise SystemExit(f"error: svd requires a 2D layer, got shape {arr.shape}")
    k = args.k
    if k < 1:
        raise SystemExit("error: --k must be >= 1")
    k = min(k, arr.shape[0], arr.shape[1])
    # economy SVD
    u, s, vt = np.linalg.svd(arr, full_matrices=False)
    u, s, vt = u[:, :k], s[:k], vt[:k, :]
    p = env_precision()
    print(f"layer={args.layer} shape={list(arr.shape)} k={k}")
    print("singular_values: " + " ".join(fmt_float(float(x), p) for x in s))
    max_print = env_max_print()
    for i in range(k):
        left = u[:, i]
        right = vt[i, :]
        if left.size + right.size > max_print:
            # print truncated heads
            head = max(8, max_print // 4)
            print(f"u[{i}][:{head}]=" + " ".join(fmt_float(float(x), p) for x in left[:head]) + " ...")
            print(f"v[{i}][:{head}]=" + " ".join(fmt_float(float(x), p) for x in right[:head]) + " ...")
        else:
            print(f"u[{i}]=" + " ".join(fmt_float(float(x), p) for x in left))
            print(f"v[{i}]=" + " ".join(fmt_float(float(x), p) for x in right))
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="inspect.py", description="Inspect model.npz weights")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("list", help="list layers")

    st = sub.add_parser("stats", help="summary stats for a layer")
    st.add_argument("layer")

    sh = sub.add_parser("show", help="print a weight slice")
    sh.add_argument("layer")
    sh.add_argument("--rows", default=None, help="row slice a:b")
    sh.add_argument("--cols", default=None, help="col slice c:d")
    sh.add_argument("--precision", type=int, default=None)

    sv = sub.add_parser("svd", help="top singular values/vectors")
    sv.add_argument("layer")
    sv.add_argument("--k", type=int, required=True)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    handlers = {
        "list": cmd_list,
        "stats": cmd_stats,
        "show": cmd_show,
        "svd": cmd_svd,
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
