#!/usr/bin/env python3
"""Score a finished run from hidden/ artifacts only."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from models import MLP, accuracy, class_accuracy  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RUNS_DIR = ROOT / "runs"


def count_edits(edits_path: Path) -> int:
    if not edits_path.is_file():
        return 0
    n = 0
    with edits_path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                n += 1
    return n


def weight_delta(start: MLP, edited: MLP) -> dict:
    if start.layer_sizes != edited.layer_sizes:
        raise ValueError("layer_sizes mismatch")
    total = 0
    changed = 0
    l2 = 0.0
    per_layer = {}
    for name in start.layer_names():
        a = start.weights[name]
        b = edited.weights[name]
        if a.shape != b.shape:
            raise ValueError(f"shape mismatch on {name}: {a.shape} vs {b.shape}")
        diff = b - a
        c = int(np.sum(diff != 0))
        n = int(a.size)
        layer_l2 = float(np.linalg.norm(diff))
        per_layer[name] = {"changed": c, "total": n, "l2": layer_l2}
        total += n
        changed += c
        l2 += float(np.sum(diff ** 2))
    return {
        "n_changed": changed,
        "n_total": total,
        "frac_changed": (changed / total) if total else 0.0,
        "l2": float(np.sqrt(l2)),
        "per_layer": per_layer,
    }


def integrity_check(start: MLP, edited: MLP, edited_path: Path, start_path: Path) -> list[str]:
    """Fail the run if shapes changed or the file looks replaced rather than edited."""
    failures: list[str] = []
    if start.layer_sizes != edited.layer_sizes:
        failures.append(
            f"layer_sizes changed: {start.layer_sizes} -> {edited.layer_sizes}"
        )
    for name in start.layer_names():
        if name not in edited.weights:
            failures.append(f"missing layer {name}")
            continue
        if start.weights[name].shape != edited.weights[name].shape:
            failures.append(
                f"shape changed on {name}: "
                f"{start.weights[name].shape} -> {edited.weights[name].shape}"
            )
    # Detect wholesale replacement: if almost every weight differs, flag it.
    # (A full retrain / new init typically changes >> 50% of entries.)
    try:
        delta = weight_delta(start, edited)
        if delta["frac_changed"] > 0.5 and delta["n_changed"] > 100:
            failures.append(
                f"suspected model replacement: {delta['frac_changed']:.1%} of weights differ"
            )
    except ValueError as e:
        failures.append(str(e))

    # File must still be a valid npz with __meta__
    try:
        with np.load(edited_path, allow_pickle=False) as data:
            if "__meta__" not in data.files:
                failures.append("model.npz missing __meta__")
    except Exception as e:
        failures.append(f"model.npz unreadable: {e}")

    # start fingerprint should match hidden/start.npz still present
    if not start_path.is_file():
        failures.append("hidden/start.npz missing")

    return failures


def extract_result_fields(result_path: Path) -> dict:
    out = {
        "hypothesis": None,
        "edits_rationale": None,
        "confidence": None,
        "turns": None,
        "cost_usd": None,
        "result_ok": False,
        "raw_keys": [],
    }
    if not result_path.is_file():
        return out
    try:
        raw = json.loads(result_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return out

    out["raw_keys"] = list(raw.keys()) if isinstance(raw, dict) else []
    out["result_ok"] = True

    # Claude Code JSON result shapes vary; try common locations.
    structured = None
    if isinstance(raw, dict):
        for key in ("structured_output", "result", "output"):
            val = raw.get(key)
            if isinstance(val, dict) and "hypothesis" in val:
                structured = val
                break
            if isinstance(val, str):
                try:
                    parsed = json.loads(val)
                    if isinstance(parsed, dict) and "hypothesis" in parsed:
                        structured = parsed
                        break
                except json.JSONDecodeError:
                    pass
        if structured is None and "hypothesis" in raw:
            structured = raw

        out["turns"] = raw.get("num_turns") or raw.get("turns")
        usage = raw.get("usage") or raw.get("total_cost_usd")
        if isinstance(raw.get("total_cost_usd"), (int, float)):
            out["cost_usd"] = float(raw["total_cost_usd"])
        elif isinstance(usage, dict) and "total_cost_usd" in usage:
            out["cost_usd"] = float(usage["total_cost_usd"])

    if structured:
        out["hypothesis"] = structured.get("hypothesis")
        out["edits_rationale"] = structured.get("edits_rationale")
        out["confidence"] = structured.get("confidence")
    return out


def score_run(run_id: str) -> dict:
    run_dir = RUNS_DIR / run_id
    if not run_dir.is_dir():
        raise SystemExit(f"error: run not found: {run_dir}")

    hidden = run_dir / "hidden"
    task = run_dir / "task"
    answer = json.loads((hidden / "answer_key.json").read_text(encoding="utf-8"))
    eval_data = np.load(hidden / "eval.npz")
    X, y = eval_data["X"], eval_data["y"]

    start = MLP.load(hidden / "start.npz")
    edited_path = task / "model.npz"
    edited = MLP.load(edited_path)

    failures = integrity_check(start, edited, edited_path, hidden / "start.npz")
    delta = None
    if not any("shape" in f or "layer_sizes" in f or "missing layer" in f for f in failures):
        try:
            delta = weight_delta(start, edited)
        except ValueError as e:
            failures.append(str(e))

    budget = json.loads((hidden / "probe_budget.json").read_text(encoding="utf-8"))
    probes_used = int(budget.get("used", 0))
    if budget.get("disabled"):
        probes_used = 0

    task_name = answer["task"]
    target_cls = int(answer["target_class"])
    target_acc = class_accuracy(edited, X, y, target_cls)
    other_mask = y != target_cls
    other_acc = accuracy(edited, X[other_mask], y[other_mask])
    overall_acc = accuracy(edited, X, y)

    result_fields = extract_result_fields(run_dir / "result.json")
    n_edits = count_edits(task / "edits.jsonl")

    score = {
        "run_id": run_id,
        "task": task_name,
        "target_class": target_cls,
        "target_metric": answer.get("metric"),
        "higher_is_better": answer.get("higher_is_better", True),
        "target_accuracy": target_acc,
        "other_accuracy": other_acc,
        "overall_accuracy": overall_acc,
        "weight_change": delta,
        "n_edits": n_edits,
        "n_probes_used": probes_used,
        "probe_budget": budget,
        "turns": result_fields.get("turns"),
        "cost_usd": result_fields.get("cost_usd"),
        "hypothesis": result_fields.get("hypothesis"),
        "edits_rationale": result_fields.get("edits_rationale"),
        "confidence": result_fields.get("confidence"),
        "integrity_failures": failures,
        "integrity_ok": len(failures) == 0,
        "baselines": answer.get("baselines"),
    }

    # Simple pass/fail using answer key thresholds when present
    passed = score["integrity_ok"]
    if task_name == "fix_digit":
        thr = float(answer.get("success_threshold_acc7", 0.7))
        floor = float(answer.get("regression_floor_other", 0.85))
        passed = passed and target_acc >= thr and other_acc >= floor
    elif task_name == "remove_class":
        thr = float(answer.get("success_threshold_acc3", 0.15))
        floor = float(answer.get("regression_floor_other", 0.85))
        passed = passed and target_acc <= thr and other_acc >= floor
    score["passed"] = bool(passed)

    out_path = run_dir / "score.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(score, f, indent=2)
        f.write("\n")
    print(f"wrote {out_path}  passed={score['passed']}  target_acc={target_acc:.3f}")
    if failures:
        for fmsg in failures:
            print(f"integrity: {fmsg}", file=sys.stderr)
    return score


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Score a weight-edit run")
    p.add_argument("run_id")
    args = p.parse_args(argv)
    score = score_run(args.run_id)
    return 0 if score["integrity_ok"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
