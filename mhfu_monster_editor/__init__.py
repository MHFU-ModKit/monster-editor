"""mhfu_monster_editor — author a ported monster end-to-end.

  * :mod:`.manifest` — `port.toml`: schema, loader, canonical writer. The single source
    of truth for a ported monster, and pure data: stdlib only.
  * :mod:`.intel` — `species/emNN.json`: what the HOST action expects per `(main,sub)`,
    joined from the offline analysers by `tools/em_intel.py`, with the provenance of
    every field. Stdlib only, like the manifest.
  * :mod:`.clips` — what is in each animation slot (CARRIED / FILLER / HOST) and whether
    a clip's NAME still points at the clip it was written for. Stdlib at import time.
  * :mod:`.align` — the HOST action's expectations and YOUR clip on one frame axis:
    cursor gates, what ends the action, effect spawns, measured dwell. Stdlib only.
  * :mod:`.validate` — the checks that need evidence (the built PAC, the action intel).
  * :mod:`.core` — a monster PAC as a render-agnostic :class:`~.core.scene.Scene`:
    skeleton, skinned geometry, textures, clips, ``pose(clip, frame)``. No GL, no UI.

The manifests themselves live at the repo root in `ports/`. ``core`` is deliberately NOT
re-exported here: it pulls in numpy and `tools/mhfu_model`, and the manifest layer is
meant to stay importable with neither. ``from mhfu_monster_editor.core import open_scene``.
"""
from .manifest import (SCHEMA, Build, Clip, Effect, Hurtbox, ManifestError, Move,
                       PortManifest, RenameClip, SetKey, Source, discover, dumps, load,
                       loads, patch, patch_file, save)

__all__ = ["SCHEMA", "Build", "Clip", "Effect", "Hurtbox", "ManifestError", "Move",
           "PortManifest", "RenameClip", "SetKey", "Source", "discover", "dumps",
           "load", "loads", "patch", "patch_file", "save"]
