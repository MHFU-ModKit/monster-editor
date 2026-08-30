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


def parse_container(blob: bytes):
    """Decode the anim container header shared by MHFU and MHP3rd.

    🔴 The header is **N × (u32 slot_count, u32 table_offset)** followed by two
    words, and ``hsize`` is nothing but the FIRST pair's table offset::

        +0x00   N × (u32 slot_count, u32 table_offset)   # N = hsize/8 - 1
        +hsize-8  u32 0
        +hsize-4  u32 data_start   (= end of the last slot table)
        +hsize    the tables, back to back, then the blocks

    Verified across every anim sub in both games: MHFU has hsize 0x18 / 0x28 /
    0x38 (2 / 4 / 6 streams; 47 / 18 / 36 files) and MHP3rd 0x20 (3 streams), and
    in every one of them ``pairs[0].offset == hsize``, the offsets chain exactly
    (``off[i+1] == off[i] + count[i]*4``) and the last table ends at ``data_start``.

    That makes both of the older readings off by one slot: ``hsize - 4`` (used
    here until 2026-08-24) lands on ``data_start``, and the reference template's
    ``hsize + 4`` skips the first stream's slot 0 *and* merges every stream into
    one flat table. Either one shifts a monster's whole moveset by a slot.

    Returns ``(pairs, word30, data_start)`` or ``None`` if the bytes do not
    validate as this layout — callers fall back to their legacy reading.
    """
    if len(blob) < 0x10:
        return None
    hsize = struct.unpack_from("<I", blob, 4)[0]
    if not (0x10 <= hsize <= 0x400 and hsize % 8 == 0 and hsize <= len(blob)):
        return None
    n = hsize // 8 - 1
    if n < 1:
        return None
    pairs = [struct.unpack_from("<2I", blob, i * 8) for i in range(n)]
    if pairs[0][1] != hsize:
        return None
    run = hsize
    for cnt, off in pairs:
        if off != run or cnt > 4096:
            return None
        run += cnt * 4
    if run > len(blob):
        return None
    word30, data_start = struct.unpack_from("<2I", blob, hsize - 8)
    if data_start != run:
        return None
    return pairs, word30, data_start


def parse(blob: bytes, stream: int = 0) -> AnimationPack:
    """Parse a P3rd-style animation pack into the data model.

    Reads the container header (:func:`parse_container`) and decodes ``stream``'s
    slot table — stream 0 (the main one) by default. Each populated slot
    (≠ 0xFFFFFFFF) points to an anim block; the nested `{flag|tag, count, size}`
    sections chain exactly. Per-anim decode is guarded so one malformed block never
    aborts the whole parse.
    """
    magic, hsize, slot_count = struct.unpack_from("<3I", blob, 0)
    hdr = parse_container(blob)
    if hdr is not None:
        slot_count, tbase = hdr[0][min(stream, len(hdr[0]) - 1)]
    else:
        tbase = hsize - 4                      # legacy fallback
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

# ---------------------------------------------------------------------------
# MHP3rd anim sub[3] — parse_p3rd()
# ---------------------------------------------------------------------------
# MHP3rd sub[3] container header (same slot-table layout as MHFU, different magic):
#   u32  anim_count      (varies per monster; NOT a fixed magic like MHFU's 0x64)
#   u32  hsize           (0x18 — same as MHFU)
#   u32  slot_count      (varies; MHFU always 100)
#   u32  first_data_off  (informational; not used by the parser)
#   ... possibly more hdr bytes ...
#   [hsize - 4]          slot offset table (slot_count * u32, 0xffffffff = empty)
#
# Per-slot block header (MHP3rd, FULLY decoded 2026-06-21 on file_04004/04005):
#   u32  bone_count
#   u32  block_size     (exact byte count of this block including this header)
#   u32  loop_flag      (0 = no loop)
#   u32  pad
#   then bone_count BONE records:
#     u16  num_channels
#     u16  record_size  (total bytes of this bone record incl this 4-byte header)
#     then num_channels CHANNEL records:
#       u16  channel_bit (SAME bits as MHFU: 0x08/0x10/0x20 rot, 0x40/0x80/0x100 loc,
#                         0x200/0x400/0x800 scl)
#       u16  keyframe_count
#       u32  channel_size (= 8 header + keyframe_count*8)
#       then keyframe_count KEYFRAMES:  <4h> value, frame, ease_in, ease_out
#                                        (IDENTICAL to the MHFU 0x64 keyframe)
#
# i.e. the SAME recursive data as MHFU, just with compact u16 bone/channel headers
# (vs MHFU's u32 + a per-bone FLAG|mask tag). The parser below produces the SAME
# Animation/BoneTrack/Channel data model the MHFU side uses, so anim_ingame's
# from_flat_anim / swap_anim_to_realmotion consume it directly (cross-game port).


def _parse_anim_p3rd(blob: bytes, ao: int, slot: int) -> Animation:
    """Parse one MHP3rd anim block (bones -> channels -> keyframes).

    🔴 EVERY COUNT HERE COMES OFF THE BLOB, so this must survive being pointed at
    a file that is not an anim pack. `bone_count` is a u32 (up to 4e9) and a
    `rec_size`/`csz` of 0 never advances the cursor — so garbage input span the
    loop appending tracks until the KERNEL killed the process. That is not a
    catchable exception: callers wrapping this in `except Exception` see no error,
    they see their whole tool die with exit 137, which reads as a machine problem.
    (Real case: `slot_catalog.py --source` handed a monster's MODEL pac instead of
    its anim pac — `file_05339` instead of `file_05341`.) A record is at least its
    own 4-byte header, so the blob's length bounds every count.
    """
    bone_count, block_size, loop_flag = struct.unpack_from("<3I", blob, ao)
    o = ao + 0x10                            # past {bone_count,block_size,loop,pad}
    tracks: list[BoneTrack] = []
    for _b in range(min(bone_count, max(0, len(blob) - o) // 4)):
        nch, rec_size = struct.unpack_from("<2H", blob, o)
        co = o + 4
        channels = []
        mask = 0
        for _c in range(min(nch, max(0, len(blob) - co) // 8)):
            bit, nkf, csz = struct.unpack_from("<HHI", blob, co)
            kfs = [Keyframe(*struct.unpack_from("<4h", blob, co + 8 + k * 8))
                   for k in range(nkf)]
            channels.append(Channel(type=bit, keyframes=kfs))   # low bits = transform bit
            mask |= bit
            if csz <= 0:
                break                        # would not advance — garbage, not a clip
            co += csz
        tracks.append(BoneTrack(tag=mask, channels=channels))   # tag low bits = mask
        if rec_size <= 0:
            break
        o += rec_size
    end = ao + min(block_size, len(blob) - ao)
    return Animation(
        slot=slot, tag=0, bone_count=bone_count, loop=loop_flag,
        loop_start=0.0, tracks=tracks, raw=blob[ao:end],
    )


def parse_p3rd(blob: bytes, stream: int = 0) -> AnimationPack:
    """Parse an MHP3rd animation sub[3] blob into the AnimationPack data model.

    MHP3rd uses the same outer container as MHFU (:func:`parse_container`) but a
    different per-slot block format. ``stream`` selects which of the container's
    tables to decode; stream 0 is the monster's main moveset.

    ⚠️ The Brute Tigrex's ``file_05250`` declares 3 streams — 70 slots, 20 slots
    and an empty one — and BOTH populated streams carry the full 43-bone rig, so
    they are two independent clip sets, not a bone partition (MHFU's 0x38 layout
    is the other way round: streams 0/2/4 partition one rig 31+9+5). Reading them
    as one 89-slot table, as the older code did, interleaved set 1 behind set 0 at
    a +69 offset and shifted set 0 by one slot.

    The returned AnimationPack carries fully-decoded tracks/channels/keyframes
    (the MHFU data model), so anim_ingame.from_flat_anim / swap_anim_to_realmotion
    consume it directly for the cross-game animation port.
    """
    if len(blob) < 0x10:
        return AnimationPack(magic=0, slot_count=0, animations=[], header=blob, raw=blob)
    count_field, hsize = struct.unpack_from("<2I", blob, 0)
    anims: list[Animation] = []
    n = 0
    hdr = parse_container(blob)
    if hdr is not None:
        n, tbase = hdr[0][min(stream, len(hdr[0]) - 1)]
        if 0 < n <= 4096 and tbase + n * 4 <= len(blob):
            table = struct.unpack_from("<%dI" % n, blob, tbase)
            for i, off in enumerate(table):
                if off == EMPTY or not (0 < off < len(blob)):
                    continue
                try:
                    anims.append(_parse_anim_p3rd(blob, off, i))
                except (struct.error, IndexError):
                    pass
    if not anims:
        # lobby fallback: slot table right after the 3-word header (no first_anim word)
        n = count_field
        tbase = 0x0c
        if 0 < n <= 1024 and tbase + n * 4 <= len(blob):
            table = struct.unpack_from("<%dI" % n, blob, tbase)
            for i, off in enumerate(table):
                if off == EMPTY or not (0 < off < len(blob)):
                    continue
                try:
                    anims.append(_parse_anim_p3rd(blob, off, i))
                except (struct.error, IndexError):
                    pass
    hdr_end = hsize if 0 < hsize <= len(blob) else min(0x18, len(blob))
    return AnimationPack(
        magic=count_field,      # raw count field (not 0x64)
        slot_count=n,           # slot-table entry count
        animations=anims,
        header=blob[:hdr_end],
        raw=blob,
    )


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
    hdr = parse_container(raw) if raw else None
    tbase = hdr[0][0][1] if hdr else hsize       # = hsize; NOT hsize-4
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
