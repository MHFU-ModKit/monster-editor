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
        channels = []
        co = o + 0x0C
        for _c in range(nch):
            ctag, nkf, csz = struct.unpack_from("<3I", a, co)
            kfs = [Keyframe(*struct.unpack_from("<4h", a, co + 0x0C + k * 8))
                   for k in range(nkf)]
            channels.append(Channel(type=ctag, keyframes=kfs))
            co += csz
        tracks.append(BoneTrack(tag=btag, channels=channels))
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


ALIGN = 4   # anim block offsets are 4-byte aligned (verified across all 49 PACs)


def _encode_channel(ch: Channel) -> bytes:
    nkf = len(ch.keyframes)
    csz = 0x0C + nkf * 8
    out = bytearray(struct.pack("<3I", ch.type, nkf, csz))
    for kf in ch.keyframes:
        out += struct.pack("<4h", _s16(kf.value), _s16(kf.frame),
                           _s16(kf.ease_in), _s16(kf.ease_out))
    return bytes(out)


def _encode_track(tr: BoneTrack) -> bytes:
    body = b"".join(_encode_channel(c) for c in tr.channels)
    bsz = 0x0C + len(body)
    return struct.pack("<3I", tr.tag, len(tr.channels), bsz) + body


def _encode_anim(a: Animation) -> bytes:
    """Serialize one anim block byte-exactly from its fields.

    Verified to reproduce the original bytes for every anim in all 49 big-monster
    PACs (no hidden padding: csz=0x0C+nkf*8, bsz=0x0C+Σch, size=0x14+Σbsz).
    """
    body = b"".join(_encode_track(t) for t in a.tracks)
    size = 0x14 + len(body)
    head = struct.pack("<3I", a.tag, len(a.tracks), size) + \
        struct.pack("<I", a.loop) + struct.pack("<f", a.loop_start)
    return head + body


def _s16(v: int) -> int:
    return max(-32768, min(32767, int(v)))


def _align(n: int) -> int:
    return (n + ALIGN - 1) & ~(ALIGN - 1)


def encode(pack: AnimationPack) -> bytes:
    """Serialize an animation pack from the data model.

    Two paths, chosen automatically (keeps the Phase 0 round-trip byte-identical
    while enabling edits):

    * **In-place** — when nothing structural changed (same slot set, no block
      changed size) AND aliased slots stay consistent: patch each unique offset's
      new block into a copy of `pack.raw`. Untouched bytes — including the opaque
      gap between the offset table and the first anim (a secondary per-slot table)
      and the file tail — are preserved verbatim, so an unedited pack returns its
      exact original bytes and a value edit touches only the affected words.
      (Slot offsets are aliased in the wild: e.g. Tigrex slots 0 and 2 share one
      clip — patching writes each unique offset once.)

    * **Rebuild** — when a block changed size, a slot was added/removed, or an
      aliased slot was edited divergently (forcing de-aliasing): preserve
      `raw[:first_anim]` verbatim (header + main table + secondary gap table),
      re-lay every populated anim 4-byte-aligned after it, and rewrite the main
      offset table. Engine-valid; not byte-identical to the source. NOTE: if a
      secondary gap table indexes anim *offsets*, a size-changing edit may need it
      refreshed too — flagged as a known limitation for structural anim edits.
    """
    raw = pack.raw or b""
    hsize = len(pack.header) if pack.header else 0x18
    tbase = hsize - 4
    blocks = {a.slot: _encode_anim(a) for a in pack.animations}
    rawlen = {a.slot: len(a.raw) for a in pack.animations}

    orig = {}
    if raw and 0 <= tbase and tbase + pack.slot_count * 4 <= len(raw):
        tbl = struct.unpack_from("<%dI" % pack.slot_count, raw, tbase)
        orig = {i: off for i, off in enumerate(tbl)
                if off not in (0, EMPTY) and off < len(raw)}

    # group slots by shared offset (alias detection)
    by_off: dict = {}
    if set(blocks) == set(orig):
        for s in blocks:
            by_off.setdefault(orig[s], []).append(s)

    can_inplace = bool(raw) and set(blocks) == set(orig)
    if can_inplace:
        for off, slots in by_off.items():
            ref = blocks[slots[0]]
            if len(ref) != rawlen[slots[0]]:               # size changed
                can_inplace = False
                break
            if any(blocks[s] != ref for s in slots):       # aliased divergent edit
                can_inplace = False
                break

    if can_inplace:
        out = bytearray(raw)
        for off, slots in by_off.items():
            out[off:off + len(blocks[slots[0]])] = blocks[slots[0]]
        return bytes(out)

    # ---- rebuild (de-aliases; preserves header + table + gap) -------------- #
    if raw and orig:
        first_off = min(orig.values())
        out = bytearray(raw[:first_off])      # header + main table + secondary gap
    else:                                     # synthetic pack, no source bytes
        first_off = tbase + pack.slot_count * 4
        head = pack.header[:tbase] if pack.header else \
            struct.pack("<3I", pack.magic, hsize, pack.slot_count)
        out = bytearray(head) + bytearray(EMPTY.to_bytes(4, "little") * pack.slot_count)
        out = out[:first_off] if len(out) >= first_off else out

    pos = first_off
    placed = []
    for slot in sorted(blocks):
        pos = _align(pos)
        placed.append((slot, pos, blocks[slot]))
        pos += len(blocks[slot])
    if len(out) < pos:
        out += bytes(pos - len(out))
    # clear table entries for slots no longer present
    for slot in orig:
        if slot not in blocks:
            struct.pack_into("<I", out, tbase + slot * 4, EMPTY)
    for slot, off, blk in placed:
        struct.pack_into("<I", out, tbase + slot * 4, off)
        out[off:off + len(blk)] = blk
    return bytes(out)
