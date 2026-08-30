"""mhfu_monster_editor — author a ported monster end-to-end.

Issue #2's slice is the **manifest layer**: `port.toml` is the single source of truth for
a ported monster, and everything else in the milestone reads it.

  * :mod:`.manifest` — schema, loader, canonical writer. Pure data, stdlib only.
  * :mod:`.validate` — the checks that need evidence (the built PAC, the action census).

The manifests themselves live at the repo root in `ports/`. No rendering, no GL and no
UI live here; issue #3 builds the render-agnostic scene on top of this layer.
"""
from .manifest import (SCHEMA, Build, Clip, Effect, Hurtbox, ManifestError, Move,
                       PortManifest, Source, discover, dumps, load, loads, save)

__all__ = ["SCHEMA", "Build", "Clip", "Effect", "Hurtbox", "ManifestError", "Move",
           "PortManifest", "Source", "discover", "dumps", "load", "loads", "save"]
