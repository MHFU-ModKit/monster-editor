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
    "repack",
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


def repack(mm: MonsterModel) -> bytes:
    """Re-encode the decoded sub-resources back into a PAC (Phase 3 write-back).

    Re-encodes the FIRST skeleton / model / anim sub from the decoded data model
    and leaves every other sub-resource (textures, a monster's second model set,
    unknown subs) byte-identical. Untouched + unedited assets reproduce the source
    file exactly (the encoders are byte-exact); edited assets re-flow offsets via
    `MonsterPac.to_bytes`. Texture (TMH) write-back is deferred to a later phase.
    """
    pac = mm.pac
    done = {"skeleton": False, "model": False, "anim": False}
    new_subs = []
    for sub in pac.subs:
        r = pac.role(sub)
        data = sub.data
        if r == "skeleton" and mm.skeleton and not done["skeleton"]:
            data = _skeleton.encode(mm.skeleton); done["skeleton"] = True
        elif r == "model" and mm.model and not done["model"]:
            data = _pmo.encode(mm.model); done["model"] = True
        elif r == "anim" and mm.anim and not done["anim"]:
            data = _anim.encode(mm.anim); done["anim"] = True
        new_subs.append(SubResource(sub.index, data))
    return MonsterPac(subs=new_subs, tail=pac.tail).to_bytes()
