"""Demote this directory on sys.path so tools/inspect.py cannot shadow stdlib inspect."""

from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent


def apply() -> None:
    tools_resolved = _TOOLS.resolve()
    tools_s = str(_TOOLS)
    new_path: list[str] = []
    for p in sys.path:
        try:
            if p and Path(p).resolve() == tools_resolved:
                continue
        except OSError:
            pass
        new_path.append(p)
    sys.path[:] = new_path
    if tools_s not in sys.path:
        sys.path.append(tools_s)


apply()
