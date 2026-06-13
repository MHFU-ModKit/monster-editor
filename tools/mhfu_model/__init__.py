"""mhfu_model — MHFU big-monster model/skeleton/animation library.

Single source of truth for the on-disk format AND the engine's constraints, shared
by the Blender addon, the CLI, and the live PRX injector. Phase 0 = read side +
byte-exact container; encoders + validators land in later phases.

Quick use:
    from mhfu_model import load_pac
    mm = load_pac("file_06134.bin")
    mm.skeleton.bone_count          # 25 (Tigrex)
    mm.anim.animations[0].tracks    # per-bone keyframe tracks
    mm.pac.to_bytes() == original   # byte-identical round-trip
"""
from __future__ import annotations

from . import anim as _anim
from . import pmo as _pmo
from . import skeleton as _skeleton
from .model import (Animation, AnimationPack, Bone, BoneTrack, Channel,
                    Keyframe, MeshGroup, Model, MonsterModel, Skeleton, Texture)
from .pac import MonsterPac, SubResource

__all__ = [
    "load_pac", "parse_pac", "MonsterPac", "SubResource",
    "MonsterModel", "Skeleton", "Bone", "Model", "MeshGroup",
    "AnimationPack", "Animation", "BoneTrack", "Channel", "Keyframe", "Texture",
]


def parse_pac(data: bytes) -> MonsterModel:
    """Parse raw PAC bytes into a decoded MonsterModel (first set: slots 0-3)."""
    pac = MonsterPac.from_bytes(data)
    mm = MonsterModel(pac=pac)
    sk = pac.find("skeleton")
    md = pac.find("model")
    tx = pac.find("texture")
    an = pac.find("anim")
    if sk:
        mm.skeleton = _skeleton.parse(sk.data)
    if md:
        mm.model = _pmo.parse(md.data)
    if tx:
        mm.texture = Texture(raw=tx.data)
    if an:
        mm.anim = _anim.parse(an.data)
    return mm


def load_pac(path: str) -> MonsterModel:
    with open(path, "rb") as f:
        return parse_pac(f.read())
