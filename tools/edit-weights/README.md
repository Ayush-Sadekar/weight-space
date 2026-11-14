# Weight-space edit harness

Headless experiment harness where **Claude Code** (`claude -p`) edits the weight matrices of small neural networks through a fixed set of tools. Claude only sees a run’s `task/` directory; evaluation data, the answer key, and the probe budget live in a sibling `hidden/` folder the agent cannot read.

v1 uses **Python 3 + numpy only** for the model (plain SGD MLP). The `.npz` layout includes a `__meta__` blob so a torch/safetensors backend can be swapped in later without changing the tool CLI.

## Layout

```
tools/edit-weights/
  README.md
  tools/            # the ONLY commands Claude may run
    inspect.py
    edit.py
    probe.py
  harness/          # code Claude never sees
    make_task.py
    run.sh
    driver.py
    score.py
    models.py
  tasks/            # templates
    fix_digit/
    remove_class/
  runs/             # gitignored; one folder per run
  tests/
```

## Setup

```bash
# from repo root (or this directory)
source ../../venv/bin/activate   # or your own venv
pip install numpy scikit-learn
```

`scikit-learn` is only used for `load_digits` (bundled 8×8 digits — no download).

## Run one task by hand

```bash
cd tools/edit-weights

# 1) Build a run
python harness/make_task.py fix_digit 0 probes_10 my_run

# 2) Optional: exercise tools yourself
cd runs/my_run/task
python ../../../tools/inspect.py list
python ../../../tools/inspect.py stats W1
python ../../../tools/edit.py set W1 7 0 0.0
python ../../../tools/probe.py --inline "$(python -c 'import json; print(json.dumps([0.0]*64))')"

# 3) Launch Claude (requires `claude` CLI)
cd ../../..   # back to edit-weights/
bash harness/run.sh my_run

# 4) Score
python harness/score.py my_run
cat runs/my_run/score.json
```

Dry-run a full driver combo (builds the task and prints the exact `claude` command, does not call Claude):

```bash
python harness/driver.py --tasks fix_digit --conditions weights_only --seeds 0 --dry-run
```

## Run a sweep

```bash
python harness/driver.py \
  --tasks fix_digit,remove_class \
  --conditions weights_only,probes_10 \
  --seeds 0,1,2 \
  --parallel 2
```

This runs `make_task → run.sh → score` for each combination, writes `runs/summary.csv`, and prints a table grouped by **task × condition** (mean ± std over seeds). Runs that already have `score.json` are skipped (resume-friendly). Pass `--no-skip` to force rebuilds.

### Environment knobs

| Var | Default | Meaning |
|-----|---------|---------|
| `WS_MODEL` | `opus` | Claude model id |
| `WS_MAX_TURNS` | `40` | `--max-turns` |
| `WS_MAX_BUDGET` | `5` | `--max-budget-usd` (estimate) |
| `WS_PRECISION` | `4` | float print precision for inspect/probe |
| `WS_MAX_PRINT` | `4096` | max values `inspect show` will print |
| `WS_TRANSCRIPT` | `0` | `1` → stream-json transcript instead of result.json |

## Conditions

Set at `make_task` time via the condition argument (also what `driver.py --conditions` sweeps):

| Condition | Probe behavior |
|-----------|----------------|
| `weights_only` | Probes disabled (`probe.py` prints that and exits) |
| `probes_10` | 10 forward-pass probes |
| `probes_50` | 50 forward-pass probes |

Precision is orthogonal: export `WS_PRECISION=2|4|8` for the Claude process / tool invocations.

## Tools

All tools must be run from inside `runs/<run_id>/task/` and operate on `model.npz` there. Documented command forms (also embedded in `task.md`) must match the Bash allowlist in `run.sh`:

```
python ../../../tools/inspect.py …
python ../../../tools/edit.py …
python ../../../tools/probe.py …
```

- **inspect** — `list` / `stats` / `show` / `svd`
- **edit** — exactly one of `set`, `add`, `scale`, `zero`, `ablate`, `save-vector` per call; appends to `edits.jsonl`
- **probe** — forward pass; decrements `../hidden/probe_budget.json` (Claude cannot write that file)

Path safety: tools refuse to read/write outside `task/` (except probe’s budget file in `hidden/`).

## Scoring

`score.py` reads **only** `hidden/` (+ the edited `task/model.npz` and `edits.jsonl`). It never trusts Claude’s narrative.

Metrics include:

- Target metric (e.g. accuracy on 7s for `fix_digit`; accuracy on class 3 for `remove_class`, where lower is better)
- Accuracy on everything else (regressions)
- Weight change vs `hidden/start.npz` (count, %, L2)
- `#edits`, `#probes used`, turns/cost from `result.json` when present
- Stated hypothesis / confidence from structured output

Integrity: fails the run if layer shapes change or the model looks wholesale-replaced (large fraction of weights differ).

## Tasks

### `fix_digit`

Train on digits, plant a bug so **7s tend to be classified as 1s** (corrupt the output row for class 7). Claude is asked to fix it with minimal weight changes.

### `remove_class`

Train normally. Claude is asked to **stop recognition of class 3** without wrecking other classes.

## Adding a new task template

1. Create `tasks/<name>/` (optional `NOTES.md` for humans — not shown to Claude).
2. Add a `build_<name>(seed, condition, run_dir)` function in `harness/make_task.py` that writes:
   - `task/model.npz`, `task/task.md`, `task/edits.jsonl`
   - `hidden/start.npz`, `hidden/eval.npz`, `hidden/answer_key.json`, `hidden/probe_budget.json`
3. Register it in `BUILDERS`.
4. Teach `score.py` how to interpret `answer_key.json` for pass/fail if needed.
5. Add a smoke test under `tests/`.

Keep `task.md` free of answer hints. Reuse the shared tool-command footer so Bash allowlist strings stay exact.

## Known limitations

- **Bash permission rules match the command text.** Claude must invoke tools with the exact `python ../../../tools/….py …` form documented in `task.md` / `run.sh`. Wrappers or different cwd break the allowlist.
- **`--max-budget-usd` is an estimate** enforced by the Claude Code CLI, not a hard accounting guarantee.
- **v1 is numpy MLPs only** (digits / synthetic). No transformers, no safetensors backend yet — the save format is intentionally metadata-tagged for a later swap.
- Probe inputs must live under `task/` (or use `--inline`); Claude cannot reach `hidden/` eval tensors directly by design.
- **`inspect.py` shares a name with the stdlib.** Tool entrypoints demote `tools/` on `sys.path` via `_pathfix.py` before importing numpy so this does not break; do not remove that bootstrap.

## Tests

```bash
cd tools/edit-weights
python -m unittest discover -s tests -v
```

Covers edit ops, probe budget, path safety, and scoring on a hand-made fixed model.
