#!/usr/bin/env python3
"""Sweep driver: tasks × conditions × seeds → make_task → run.sh → score."""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HARNESS = Path(__file__).resolve().parent
RUNS_DIR = ROOT / "runs"
sys.path.insert(0, str(HARNESS))

from make_task import make_task  # noqa: E402


def parse_list(s: str) -> list[str]:
    return [x.strip() for x in s.split(",") if x.strip()]


def run_id_for(task: str, condition: str, seed: int) -> str:
    return f"{task}__{condition}__seed{seed}"


def claude_command_preview(run_id: str) -> str:
    """Exact command run.sh would invoke (for dry-run)."""
    model = os.environ.get("WS_MODEL", "opus")
    max_turns = os.environ.get("WS_MAX_TURNS", "40")
    max_budget = os.environ.get("WS_MAX_BUDGET", "5")
    schema = (
        '{"type":"object","properties":{"hypothesis":{"type":"string"},'
        '"edits_rationale":{"type":"string"},"confidence":{"type":"number"}},'
        '"required":["hypothesis","edits_rationale"]}'
    )
    task_dir = RUNS_DIR / run_id / "task"
    if (os.environ.get("WS_TRANSCRIPT", "0") == "1"):
        out = f"> {RUNS_DIR / run_id / 'transcript.jsonl'}"
        fmt = "--output-format stream-json --verbose"
    else:
        out = f"> {RUNS_DIR / run_id / 'result.json'}"
        fmt = "--output-format json"
    return (
        f'cd {task_dir} && claude -p "$(cat task.md)" \\\n'
        f'  --bare \\\n'
        f'  --model "{model}" \\\n'
        f'  --tools "Bash,Read" \\\n'
        f'  --allowedTools "Bash(python ../../../tools/inspect.py *)" '
        f'"Bash(python ../../../tools/edit.py *)" '
        f'"Bash(python ../../../tools/probe.py *)" "Read(./**)" \\\n'
        f'  --permission-mode dontAsk \\\n'
        f'  --max-turns "{max_turns}" \\\n'
        f'  --max-budget-usd "{max_budget}" \\\n'
        f'  {fmt} \\\n'
        f"  --json-schema '{schema}' \\\n"
        f"  {out}"
    )


def score_run_id(run_id: str) -> dict:
    from score import score_run

    return score_run(run_id)


def execute_one(
    task: str,
    condition: str,
    seed: int,
    *,
    dry_run: bool,
    skip_existing: bool,
) -> dict:
    rid = run_id_for(task, condition, seed)
    score_path = RUNS_DIR / rid / "score.json"
    if skip_existing and score_path.is_file():
        print(f"skip {rid} (score.json exists)")
        return json.loads(score_path.read_text(encoding="utf-8"))

    make_task(task, seed, condition, rid)
    cmd = claude_command_preview(rid)
    if dry_run:
        print("=" * 60)
        print(f"DRY-RUN {rid}")
        print(cmd)
        print("=" * 60)
        return {"run_id": rid, "dry_run": True, "command": cmd}

    run_sh = HARNESS / "run.sh"
    env = os.environ.copy()
    # Precision variants can be set per-condition externally via WS_PRECISION
    subprocess.run(["bash", str(run_sh), rid], check=True, env=env, cwd=str(ROOT))
    return score_run_id(rid)


def summarize(scores: list[dict], out_csv: Path) -> None:
    if not scores:
        print("no scores")
        return
    fieldnames = [
        "run_id",
        "task",
        "condition",
        "seed",
        "passed",
        "target_accuracy",
        "other_accuracy",
        "n_edits",
        "n_probes_used",
        "frac_changed",
        "l2",
        "turns",
        "cost_usd",
        "confidence",
        "integrity_ok",
    ]
    rows = []
    for s in scores:
        if s.get("dry_run"):
            continue
        rid = s["run_id"]
        parts = rid.split("__")
        condition = parts[1] if len(parts) >= 2 else ""
        seed_s = parts[2].replace("seed", "") if len(parts) >= 3 else ""
        wc = s.get("weight_change") or {}
        rows.append(
            {
                "run_id": rid,
                "task": s.get("task"),
                "condition": condition,
                "seed": seed_s,
                "passed": s.get("passed"),
                "target_accuracy": s.get("target_accuracy"),
                "other_accuracy": s.get("other_accuracy"),
                "n_edits": s.get("n_edits"),
                "n_probes_used": s.get("n_probes_used"),
                "frac_changed": wc.get("frac_changed"),
                "l2": wc.get("l2"),
                "turns": s.get("turns"),
                "cost_usd": s.get("cost_usd"),
                "confidence": s.get("confidence"),
                "integrity_ok": s.get("integrity_ok"),
            }
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"wrote {out_csv}")

    # Grouped table: task × condition
    from collections import defaultdict
    import math

    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for r in rows:
        groups[(r["task"], r["condition"])].append(r)

    def mean_std(vals: list[float]) -> str:
        vals = [v for v in vals if v is not None and not (isinstance(v, float) and math.isnan(v))]
        if not vals:
            return "n/a"
        m = sum(vals) / len(vals)
        if len(vals) == 1:
            return f"{m:.3f}"
        var = sum((x - m) ** 2 for x in vals) / (len(vals) - 1)
        return f"{m:.3f}±{var**0.5:.3f}"

    print("\nSummary (mean ± std over seeds)")
    print(f"{'task':<14} {'condition':<14} {'target_acc':<14} {'other_acc':<14} {'passed':<10} {'edits':<10}")
    for (task, cond), items in sorted(groups.items()):
        print(
            f"{task:<14} {cond:<14} "
            f"{mean_std([float(i['target_accuracy']) for i in items if i['target_accuracy'] is not None]):<14} "
            f"{mean_std([float(i['other_accuracy']) for i in items if i['other_accuracy'] is not None]):<14} "
            f"{mean_std([1.0 if i['passed'] else 0.0 for i in items]):<10} "
            f"{mean_std([float(i['n_edits']) for i in items if i['n_edits'] is not None]):<10}"
        )


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Weight-space edit experiment driver")
    p.add_argument("--tasks", default="fix_digit,remove_class")
    p.add_argument("--conditions", default="weights_only,probes_10")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--parallel", type=int, default=1)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--no-skip",
        action="store_true",
        help="rebuild/rerun even if score.json exists",
    )
    args = p.parse_args(argv)

    tasks = parse_list(args.tasks)
    conditions = parse_list(args.conditions)
    seeds = [int(x) for x in parse_list(args.seeds)]
    skip_existing = not args.no_skip and not args.dry_run

    jobs = [(t, c, s) for t in tasks for c in conditions for s in seeds]
    results: list[dict] = []

    if args.parallel <= 1 or args.dry_run:
        for t, c, s in jobs:
            results.append(
                execute_one(t, c, s, dry_run=args.dry_run, skip_existing=skip_existing)
            )
    else:
        with ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = {
                ex.submit(
                    execute_one, t, c, s, dry_run=False, skip_existing=skip_existing
                ): (t, c, s)
                for t, c, s in jobs
            }
            for fut in as_completed(futs):
                results.append(fut.result())

    if not args.dry_run:
        summarize(results, RUNS_DIR / "summary.csv")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
