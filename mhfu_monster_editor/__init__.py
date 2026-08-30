"""mhfu_monster_editor — author a ported monster end-to-end.

  * :mod:`.manifest` — `port.toml`: schema, loader, canonical writer. The single source
    of truth for a ported monster, and pure data: stdlib only.
  * :mod:`.validate` — the checks that need evidence (the built PAC, the action census).
  * :mod:`.core` — a monster PAC as a render-agnostic :class:`~.core.scene.Scene`:
    skeleton, skinned geometry, textures, clips, ``pose(clip, frame)``. No GL, no UI.

The manifests themselves live at the repo root in `ports/`. ``core`` is deliberately NOT
re-exported here: it pulls in numpy and `tools/mhfu_model`, and the manifest layer is
meant to stay importable with neither. ``from mhfu_monster_editor.core import open_scene``.
"""
from .manifest import (SCHEMA, Build, Clip, Effect, Hurtbox, ManifestError, Move,
                       PortManifest, Source, discover, dumps, load, loads, save)

__all__ = ["SCHEMA", "Build", "Clip", "Effect", "Hurtbox", "ManifestError", "Move",
           "PortManifest", "Source", "discover", "dumps", "load", "loads", "save"]
