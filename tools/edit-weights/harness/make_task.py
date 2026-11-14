#!/usr/bin/env python3
"""Build a run directory from a task template."""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(Path(__file__).resolve().parent))

from models import (  # noqa: E402
    MLP,
    accuracy,
    class_accuracy,
    train_digits_mlp,
)

TASKS_DIR = ROOT / "tasks"
RUNS_DIR = ROOT / "runs"

TOOL_CMD = {
    "inspect": "python ../../../tools/inspect.py",
    "edit": "python ../../../tools/edit.py",
    "probe": "python ../../../tools/probe.py",
}

TASK_MD_FOOTER = """
## Tools (exact command forms — match these literally)

You may only use these Bash commands (plus Read on files under the current directory):

```
{inspect} list
{inspect} stats <layer>
{inspect} show <layer> [--rows a:b] [--cols c:d] [--precision N]
{inspect} svd <layer> --k N
{edit} set <layer> <i> <j> <value>
{edit} add <layer> <i> <j> <delta>
{edit} scale <layer> [--rows a:b] [--cols c:d] <factor>
{edit} zero <layer> [--rows a:b] [--cols c:d]
{edit} ablate <layer> <vector.npy> [--side out|in]
{edit} save-vector <name> <comma-separated values>
{probe} <input.json>
{probe} --inline "[...]"
```

Vectors created with save-vector land in `vectors/<name>.npy`.
Change as few weights as possible. Work only inside this directory.
""".strip()


def probe_budget_for_condition(condition: str) -> dict:
    if condition == "weights_only":
        return {"disabled": True, "remaining": 0, "used": 0, "initial": 0}
    if condition == "probes_10":
        return {"disabled": False, "remaining": 10, "used": 0, "initial": 10}
    if condition == "probes_50":
        return {"disabled": False, "remaining": 50, "used": 0, "initial": 50}
    raise SystemExit(f"error: unknown condition {condition!r}")


def write_probe_budget(hidden: Path, condition: str) -> None:
    data = probe_budget_for_condition(condition)
    with (hidden / "probe_budget.json").open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
        f.write("\n")


def save_eval(hidden: Path, X: np.ndarray, y: np.ndarray) -> None:
    np.savez(hidden / "eval.npz", X=X, y=y)


def save_answer_key(hidden: Path, payload: dict) -> None:
    with (hidden / "answer_key.json").open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
        f.write("\n")


def footer() -> str:
    return TASK_MD_FOOTER.format(**TOOL_CMD)


def plant_7_as_1_bug(model: MLP) -> dict:
    """Corrupt the output layer so digit 7s tend to be predicted as 1.

    Fold class-7's output weights into class 1, then kill class 7's row/bias
    so the evidence that used to support 7 now votes for 1.
    """
    out_i = model.n_layers - 1
    W = model.weights[f"W{out_i}"]
    b = model.weights[f"b{out_i}"]
    if W.shape[0] < 8:
        raise RuntimeError("output layer too small for digit classes")
    before = {
        "W7": W[7].copy().tolist(),
        "b7": float(b[7]),
        "W1": W[1].copy().tolist(),
        "b1": float(b[1]),
    }
    W[1] = W[1] + W[7]
    b[1] = float(b[1] + b[7])
    W[7] = 0.0
    b[7] = -10.0
    return {
        "bug": "7_classified_as_1",
        "method": "fold_row_7_into_1_then_zero_7",
        "output_layer": f"W{out_i}",
        "restored_hint": before,  # for answer key only
    }


def build_fix_digit(seed: int, condition: str, run_dir: Path) -> None:
    task = run_dir / "task"
    hidden = run_dir / "hidden"
    task.mkdir(parents=True)
    hidden.mkdir(parents=True)

    model, data = train_digits_mlp(seed=seed, hidden=(32,), epochs=80, lr=0.25)
    Xev, yev = data["X_eval"], data["y_eval"]

    clean = model.copy()
    bug_meta = plant_7_as_1_bug(model)

    clean.save(hidden / "original.npz")  # pre-bug reference for "ideal" is clean
    # For scoring weight-change, original = buggy starting point Claude receives
    model.save(hidden / "start.npz")
    model.save(task / "model.npz")
    save_eval(hidden, Xev, yev)
    write_probe_budget(hidden, condition)

    acc7_bug = class_accuracy(model, Xev, yev, 7)
    acc7_clean = class_accuracy(clean, Xev, yev, 7)
    acc_else_bug = accuracy(model, Xev[yev != 7], yev[yev != 7])
    conf_as_1 = float(
        (model.predict(Xev[yev == 7]) == 1).mean()
    ) if np.any(yev == 7) else float("nan")

    save_answer_key(
        hidden,
        {
            "task": "fix_digit",
            "target_class": 7,
            "confused_as": 1,
            "metric": "accuracy_on_class_7",
            "higher_is_better": True,
            "bug": bug_meta,
            "baselines": {
                "acc_class_7_buggy": acc7_bug,
                "acc_class_7_clean": acc7_clean,
                "acc_other_buggy": acc_else_bug,
                "frac_7_predicted_as_1": conf_as_1,
            },
            "success_threshold_acc7": 0.7,
            "regression_floor_other": 0.85,
        },
    )

    # Also keep clean model for analysis
    clean.save(hidden / "clean.npz")

    (task / "edits.jsonl").write_text("", encoding="utf-8")
    (task / "vectors").mkdir(exist_ok=True)

    body = f"""# Task: fix a digit classifier

You are given a small neural network (`model.npz`) trained to classify 8x8
handwritten digits (inputs are length-64 vectors with values in [0, 1];
outputs are 10 logits for digits 0–9).

Something is wrong: **many 7s are being classified as 1s**. Your job is to
**fix the model by editing its weights** so that 7s are recognized again,
while changing as few weights as possible and avoiding collateral damage to
other digits.

Use inspect / edit / probe (if available) to diagnose and repair.
Do not retrain. Do not replace the file with a new model — only edit weights
in place via the edit tool.

When finished, return JSON with your hypothesis, edits_rationale, and confidence.

{footer()}
"""
    (task / "task.md").write_text(body, encoding="utf-8")

    # Template copy of any extra assets
    tmpl = TASKS_DIR / "fix_digit"
    if (tmpl / "NOTES.md").is_file():
        shutil.copy(tmpl / "NOTES.md", hidden / "NOTES.md")


def build_remove_class(seed: int, condition: str, run_dir: Path) -> None:
    task = run_dir / "task"
    hidden = run_dir / "hidden"
    task.mkdir(parents=True)
    hidden.mkdir(parents=True)

    model, data = train_digits_mlp(seed=seed, hidden=(32,), epochs=80, lr=0.25)
    Xev, yev = data["X_eval"], data["y_eval"]

    model.save(hidden / "original.npz")
    model.save(hidden / "start.npz")
    model.save(task / "model.npz")
    save_eval(hidden, Xev, yev)
    write_probe_budget(hidden, condition)

    acc3 = class_accuracy(model, Xev, yev, 3)
    acc_other = accuracy(model, Xev[yev != 3], yev[yev != 3])

    save_answer_key(
        hidden,
        {
            "task": "remove_class",
            "target_class": 3,
            "metric": "accuracy_on_class_3",
            "higher_is_better": False,
            "baselines": {
                "acc_class_3": acc3,
                "acc_other": acc_other,
            },
            "success_threshold_acc3": 0.15,
            "regression_floor_other": 0.85,
        },
    )

    (task / "edits.jsonl").write_text("", encoding="utf-8")
    (task / "vectors").mkdir(exist_ok=True)

    body = f"""# Task: remove recognition of one class

You are given a small neural network (`model.npz`) trained to classify 8x8
handwritten digits (inputs are length-64 vectors with values in [0, 1];
outputs are 10 logits for digits 0–9).

Your job: **make the model stop recognizing digit 3**. Predictions on inputs
that are 3s should preferably go to some other class. Do this by editing
weights only. Minimize damage to accuracy on the other digits, and change as
few weights as possible.

Use inspect / edit / probe (if available) to diagnose and intervene.
Do not retrain. Do not replace the file with a new model — only edit weights
in place via the edit tool.

When finished, return JSON with your hypothesis, edits_rationale, and confidence.

{footer()}
"""
    (task / "task.md").write_text(body, encoding="utf-8")


BUILDERS = {
    "fix_digit": build_fix_digit,
    "remove_class": build_remove_class,
}


def make_task(template: str, seed: int, condition: str, run_id: str) -> Path:
    if template not in BUILDERS:
        raise SystemExit(f"error: unknown template {template!r}; known: {sorted(BUILDERS)}")
    probe_budget_for_condition(condition)  # validate early

    run_dir = RUNS_DIR / run_id
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    meta = {
        "run_id": run_id,
        "template": template,
        "seed": seed,
        "condition": condition,
    }
    with (run_dir / "meta.json").open("w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
        f.write("\n")

    BUILDERS[template](seed, condition, run_dir)
    print(f"created {run_dir}")
    return run_dir


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Build a weight-edit task run")
    p.add_argument("template", choices=sorted(BUILDERS))
    p.add_argument("seed", type=int)
    p.add_argument("condition", choices=("weights_only", "probes_10", "probes_50"))
    p.add_argument("run_id")
    args = p.parse_args(argv)
    make_task(args.template, args.seed, args.condition, args.run_id)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
