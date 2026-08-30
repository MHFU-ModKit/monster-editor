"""`mhfu_monster_editor.core` — a monster PAC as a render-agnostic scene.

Issue #3's slice. Everything the viewer needs already existed in `tools/mhfu_model/`;
this assembles it into one object and does **nothing else** — no GL, no UI, no Blender
— so it stays testable headless and can be driven from either direction: the app
(issues #6/#13), the Blender addon, or a CLI.

  * :mod:`.scene` — :class:`Scene`: bind skeleton + parent array, per-vertex
    ``(bone, weight)`` palette, mesh groups with UVs and a texture binding, decoded
    RGBA textures, the clip list, and ``pose(clip, frame)``.
  * :mod:`.pose` — the FK and blend skinning, vectorised. `tools/mhfu_model/stretch.py`
    remains the ORACLE; `tests/test_core_pose.py` pins this against it.

    python -m mhfu_monster_editor.core <pac|port.toml>
"""
from .pose import Curves, Pose, Rig, SkinBinding, effective_parents, sample
from .scene import (BODY_STREAMS, Clip, MeshGroup, Scene, SceneError, TextureImage,
                    detect_game, open_scene, stream_joint_bases)

__all__ = ["BODY_STREAMS", "Clip", "Curves", "MeshGroup", "Pose", "Rig", "Scene",
           "SceneError", "SkinBinding", "TextureImage", "detect_game",
           "effective_parents", "open_scene", "sample", "stream_joint_bases"]
