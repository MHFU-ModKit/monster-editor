"""Animation pack (PAC sub-3, P3rd-style) <-> data model.

Decode logic ported verbatim from tools/anim.py (validated 2026-06-13: Tigrex
file_06134 = 22 anims, 24 bone records each, size-chains exact). See
docs/ANIMATION_FORMAT.md.

`encode()` is a lossless passthrough until the Phase 3 animation encoder lands.
"""
from __future__ import annotations

import struct

from .model import Animation, AnimationPack, BoneTrack, Channel, Keyframe

FLAG = 0x80000000
EMPTY = 0xFFFFFFFF

# P3rd transform-channel bits (low half of a bone record's tag)
CHANNEL_BITS = [
    (0x008, "rotX"), (0x010, "rotY"), (0x020, "rotZ"),
    (0x040, "locX"), (0x080, "locY"), (0x100, "locZ"),
    (0x200, "sclX"), (0x400, "sclY"), (0x800, "sclZ"),
]


def _parse_anim(a: bytes, ao: int, slot: int) -> Animation:
    tag, bone_count, size = struct.unpack_from("<3I", a, ao)
    loop = struct.unpack_from("<I", a, ao + 0x0C)[0]
    loop_start = struct.unpack_from("<f", a, ao + 0x10)[0]
    tracks = []
    o = ao + 0x14
    for _ in range(bone_count):
        btag, nch, bsz = struct.unpack_from("<3I", a, o)
        mask = btag & 0xFFFF
        channels = []
        co = o + 0x0C
        for _c in range(nch):
            ctag, nkf, csz = struct.unpack_from("<3I", a, co)
            kfs = [Keyframe(*struct.unpack_from("<4h", a, co + 0x0C + k * 8))
                   for k in range(nkf)]
            channels.append(Channel(type=ctag, keyframes=kfs))
            co += csz
        tracks.append(BoneTrack(mask=mask, channels=channels))
        o += bsz
    return Animation(
        slot=slot, tag=tag, bone_count=bone_count, loop=loop,
        loop_start=loop_start, tracks=tracks, raw=a[ao:ao + size],
    )


def parse(blob: bytes) -> AnimationPack:
    """Parse a P3rd-style animation pack into the data model.

    Header: u32 magic(0x64) ; u32 hsize ; u32 slot_count ; ... ; the offset table
    sits at **hsize - 4** (validated on all 49 big-monster PACs: single-set
    hsize=0x18 → table @0x14; dual-model-set variant hsize=0x38 → table @0x34).
    Each populated slot (≠ 0xFFFFFFFF) points to an anim block; the nested
    `{flag|tag, count, size}` sections chain exactly. Per-anim decode is guarded so
    one malformed block never aborts the whole parse.
    """
    magic, hsize, slot_count = struct.unpack_from("<3I", blob, 0)
    tbase = hsize - 4
    anims = []
    if 0 <= tbase and tbase + slot_count * 4 <= len(blob):
        table = struct.unpack_from("<%dI" % slot_count, blob, tbase)
        for i, off in enumerate(table):
            if off == EMPTY or not (0 < off < len(blob)):
                continue
            try:
                anims.append(_parse_anim(blob, off, i))
            except (struct.error, IndexError):
                pass                        # odd block — keep raw, skip
    return AnimationPack(
        magic=magic, slot_count=slot_count, animations=anims,
        header=blob[:hsize] if 0 < hsize <= len(blob) else blob[:0x18], raw=blob,
    )


def encode(pack: AnimationPack) -> bytes:
    """Lossless passthrough (Phase 0). Phase 3 will rebuild from fields."""
    return pack.raw
