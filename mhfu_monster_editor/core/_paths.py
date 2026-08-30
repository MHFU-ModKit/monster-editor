"""Put `tools/` on `sys.path` so `mhfu_model` imports.

`manifest.py` is stdlib-only by rule; `validate.py` and this package are allowed to
reach into `tools/`, and — like `validate.py` — they do their own path setup rather
than making the caller arrange it.
"""
from __future__ import annotations

import sys
from pathlib import Path

_TOOLS = Path(__file__).resolve().parent.parent.parent / "tools"


def tools_on_path() -> str:
    """Prepend the repo's `tools/` to `sys.path` (idempotent). Returns the path."""
    p = str(_TOOLS)
    if p not in sys.path:
        sys.path.insert(0, p)
    return p
