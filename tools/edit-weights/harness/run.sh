#!/usr/bin/env bash
# Run Claude Code against one prepared task directory.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
RUN_ID="${1:?usage: run.sh <run_id>}"
RUN_DIR="$ROOT/runs/$RUN_ID"
TASK_DIR="$RUN_DIR/task"

if [[ ! -d "$TASK_DIR" ]]; then
  echo "error: missing task dir: $TASK_DIR" >&2
  exit 1
fi

# Symlink tools next to task.md so allowedTools paths stay simple if desired.
# Primary documented form uses ../../../tools/ from task/.
TOOLS_LINK="$TASK_DIR/tools"
if [[ ! -e "$TOOLS_LINK" ]]; then
  ln -s ../../../tools "$TOOLS_LINK"
fi

cd "$TASK_DIR"

PROMPT="$(cat task.md)"
MODEL="${WS_MODEL:-opus}"
MAX_TURNS="${WS_MAX_TURNS:-40}"
MAX_BUDGET="${WS_MAX_BUDGET:-5}"
SCHEMA='{"type":"object","properties":{"hypothesis":{"type":"string"},"edits_rationale":{"type":"string"},"confidence":{"type":"number"}},"required":["hypothesis","edits_rationale"]}'

ALLOWED=(
  --allowedTools
  "Bash(python ../../../tools/inspect.py *)"
  "Bash(python ../../../tools/edit.py *)"
  "Bash(python ../../../tools/probe.py *)"
  "Read(./**)"
)

COMMON=(
  -p "$PROMPT"
  --bare
  --model "$MODEL"
  --tools "Bash,Read"
  "${ALLOWED[@]}"
  --permission-mode dontAsk
  --max-turns "$MAX_TURNS"
  --max-budget-usd "$MAX_BUDGET"
  --json-schema "$SCHEMA"
)

if [[ "${WS_TRANSCRIPT:-0}" == "1" ]]; then
  claude "${COMMON[@]}" \
    --output-format stream-json \
    --verbose \
    > "$RUN_DIR/transcript.jsonl"
  # Best-effort: also leave a stub result.json if absent
  if [[ ! -f "$RUN_DIR/result.json" ]]; then
    echo '{"note":"transcript mode; see transcript.jsonl"}' > "$RUN_DIR/result.json"
  fi
else
  claude "${COMMON[@]}" \
    --output-format json \
    > "$RUN_DIR/result.json"
fi

echo "finished run $RUN_ID"
