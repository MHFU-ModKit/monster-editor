"""`mhfu_monster_editor.ui` — the imgui layer, and the only macOS-only part.

The third of issue #5's three layers: `core` (no GL) -> `render` (GL, with or without a
window) -> **`ui`** (imgui). Nothing below this line imports imgui, which is what keeps
a render assertion reachable from a machine with no display.

Importing this module pulls in `imgui-bundle`; :mod:`..render` does not, and neither
does :mod:`..core`.
"""
from .app import EditorApp, run

__all__ = ["EditorApp", "run"]
