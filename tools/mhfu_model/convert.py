"""Blender-independent conversion helpers (dequantization, bone tree, channels).

Kept free of `bpy` so the math is unit-testable headlessly; the Blender addon
(`blender_mhfu/`) is a thin glue layer that calls these and creates scene data.

Conventions (P3rd-style, see docs/ANIMATION_FORMAT.md and the sibling addon
Kurogami2134/blender_p3rd_anim): rotation 4096 = 90 deg, location 16 = 1.0,
scale 256 = 1.0. Animation channels carry a bone's LOCAL transform per frame;
the skeleton bind-pose is the rest transform.
"""
from __future__ import annotations

import math
from typing import Dict, List, Tuple

from .anim import CHANNEL_BITS
from .model import Animation, Skeleton

# channel bit -> (kind, axis_index)  (axis 0=X,1=Y,2=Z)
_KIND = {
    0x008: ("rot", 0), 0x010: ("rot", 1), 0x020: ("rot", 2),
    0x040: ("loc", 0), 0x080: ("loc", 1), 0x100: ("loc", 2),
    0x200: ("scl", 0), 0x400: ("scl", 1), 0x800: ("scl", 2),
}

ROT_UNIT = 4096.0    # = 90 deg
LOC_UNIT = 16.0      # = 1.0
SCL_UNIT = 256.0     # = 1.0


def channel_kind(channel_type: int) -> Tuple[str, int]:
    """Map a channel record's tag to (kind, axis). kind in {rot,loc,scl}."""
    bit = channel_type & 0xFFFF
    if bit in _KIND:
        return _KIND[bit]
    # some packs carry the bit in the low byte only / with the FLAG set; mask it
    for b, ka in _KIND.items():
        if bit & b:
            return ka
    return ("unknown", 0)


def dequantize(kind: str, raw: int) -> float:
    """Raw s16 -> engine float. Rotation is returned in RADIANS."""
    if kind == "rot":
        return math.radians(raw * 90.0 / ROT_UNIT)
    if kind == "loc":
        return raw / LOC_UNIT
    if kind == "scl":
        return raw / SCL_UNIT
    return float(raw)


def quantize(kind: str, value: float) -> int:
    """Inverse of dequantize (for Phase 3 encoders); clamps to s16."""
    if kind == "rot":
        q = round(math.degrees(value) * ROT_UNIT / 90.0)
    elif kind == "loc":
        q = round(value * LOC_UNIT)
    elif kind == "scl":
        q = round(value * SCL_UNIT)
    else:
        q = round(value)
    return max(-32768, min(32767, q))


def bone_local_offsets(skel: Skeleton) -> Dict[int, Tuple[float, float, float]]:
    """Return each bone's bind position as stored (treated as LOCAL offset from
    its parent — matches the engine's FK accumulation in Joint::update)."""
    return {b.index: tuple(b.bind_pos) for b in skel.bones}


def bone_world_positions(skel: Skeleton) -> Dict[int, Tuple[float, float, float]]:
    """Accumulate local bind offsets down the parent tree -> world rest positions.

    Used to place Blender bone heads. A bone's head = parent head + its local
    bind_pos; roots start at their own bind_pos.
    """
    by_index = {b.index: b for b in skel.bones}
    world: Dict[int, Tuple[float, float, float]] = {}

    def resolve(i: int) -> Tuple[float, float, float]:
        if i in world:
            return world[i]
        b = by_index[i]
        lx, ly, lz = b.bind_pos
        if b.parent == -1 or b.parent not in by_index:
            w = (lx, ly, lz)
        else:
            px, py, pz = resolve(b.parent)
            w = (px + lx, py + ly, pz + lz)
        world[i] = w
        return w

    for b in skel.bones:
        resolve(b.index)
    return world


def children_of(skel: Skeleton) -> Dict[int, List[int]]:
    out: Dict[int, List[int]] = {b.index: [] for b in skel.bones}
    for b in skel.bones:
        if b.parent != -1 and b.parent in out:
            out[b.parent].append(b.index)
    return out


def animation_fcurves(anim: Animation):
    """Flatten an Animation into per-(bone, kind, axis) keyframe lists.

    Yields tuples: (bone_index, kind, axis, [(frame, value_float), ...]).
    `bone_index` is the track's position == skeleton bone index. Values are
    dequantized (rotation in radians). Frames come straight from the keyframe
    `frame` field (Blender frame numbers).
    """
    for bone_index, track in enumerate(anim.tracks):
        for ch in track.channels:
            kind, axis = channel_kind(ch.type)
            if kind == "unknown":
                continue
            pts = [(kf.frame, dequantize(kind, kf.value)) for kf in ch.keyframes]
            if pts:
                yield bone_index, kind, axis, pts
