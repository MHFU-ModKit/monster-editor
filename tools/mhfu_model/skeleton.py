"""Skeleton (PAC sub-0, 0xC0000000 blob) <-> data model.

Decode logic ported verbatim from tools/skeleton.py (validated 2026-06-13 against
engine builder EU 0x088dc40c and the iOS m2jean format). See docs/ANIMATION_FORMAT.md.

`encode()` is a lossless passthrough (returns the original bytes) until the Phase 3
skeleton encoder replaces it; this keeps the round-trip byte-identical today.
"""
from __future__ import annotations

import struct

from .model import Bone, Skeleton

MAGIC = 0xC0000000
SECTION_MAGIC = 0x40000001
HDR_SIZE = 0x1C


def parse(blob: bytes) -> Skeleton:
    magic, bone_count, total_size = struct.unpack_from("<3I", blob, 0)
    if magic != MAGIC:
        raise ValueError("not a 0xC0000000 skeleton (magic=0x%08x)" % magic)
    bones = []
    o = HDR_SIZE
    for i in range(bone_count):
        if o + 0x1C > len(blob):
            break
        smag, flag, ssize = struct.unpack_from("<3I", blob, o)
        idx, parent, child, sibling = struct.unpack_from("<4i", blob, o + 0x0C)
        scale = struct.unpack_from("<3f", blob, o + 0x1C)
        rotation = struct.unpack_from("<3f", blob, o + 0x2C)
        position = struct.unpack_from("<3f", blob, o + 0x3C)
        section = blob[o:o + ssize] if (0 < ssize <= 0x400) else blob[o:o + 0x10C]
        bones.append(Bone(
            index=idx, parent=parent, child=child, sibling=sibling,
            bind_scale=scale, bind_rot=rotation, bind_pos=position,
            flag=flag, section_size=ssize, raw=section,
        ))
        if smag != SECTION_MAGIC or ssize == 0 or ssize > 0x400:
            break
        o += ssize
    return Skeleton(
        bone_count=bone_count, total_size=total_size, bones=bones,
        header=blob[:HDR_SIZE], raw=blob,
    )


def parse_p3rd(blob: bytes) -> Skeleton:
    """Parse a 0x80000000 MHP3rd skeleton blob into the Skeleton data model.

    Delegates to skeleton_p3rd.parse() which handles:
    - Section magic 0x40000002 (in addition to 0x40000001)
    - Optional extra 4-byte header word before bone sections (lobby PACs)

    The returned Skeleton is compatible with the Blender armature builder
    and exporter — same dataclass, same field layout.
    """
    from .skeleton_p3rd import parse as _p3rd
    return _p3rd(blob)


def _encode_bone(b: Bone) -> bytes:
    """Patch a bone's editable fields back into its raw section (same size).

    Keeps every non-field byte (the implementation-private matrix/aux region
    0x48..section_size) verbatim, so an unedited bone re-emits byte-identically
    and a bind-pose/tree edit changes only the intended words.
    """
    sec = bytearray(b.raw)
    if len(sec) < 0x48:
        return bytes(sec)                       # terminator / short final section
    struct.pack_into("<I", sec, 0x04, b.flag)
    struct.pack_into("<4i", sec, 0x0C, b.index, b.parent, b.child, b.sibling)
    struct.pack_into("<3f", sec, 0x1C, *b.bind_scale)
    struct.pack_into("<3f", sec, 0x2C, *b.bind_rot)
    struct.pack_into("<3f", sec, 0x3C, *b.bind_pos)
    return bytes(sec)


def assign_stream_ids(skel: Skeleton, split) -> Skeleton:
    """Set each bone's stream-id (`bone+0x50`, u16 → `joint+0x114`) so the anim FK
    partitions all bones into streams without overrunning.

    RE (2026-06-28, disasm `0x0886010c lw a2,0x50(a1)` → `0x08860110 sh a2,0x114(a0)`
    + native Tigrex `file_06185`): the stream-id is `bone+0x50`; a valid partition is
    **contiguous runs by bone index**. Native Tigrex = `{0:31, 1:9, 2:5, 3:3}`. An
    MHP3rd source skeleton ships `+0x50`=0 on every bone (all stream 0 → FK overrun
    crash), so a ported skeleton MUST be assigned before injection.

    `split` = per-stream bone counts (e.g. [31, 9, 5, 3]); must match the anim's
    stream partition. Patches each bone's raw section at +0x50 in place. Returns skel.
    """
    runs = []
    for sid, n in enumerate(split):
        runs += [sid] * n
    if len(runs) < len(skel.bones):
        runs += [len(split) - 1] * (len(skel.bones) - len(runs))   # trailing -> last stream
    for b, sid in zip(skel.bones, runs):
        if len(b.raw) >= 0x52:
            sec = bytearray(b.raw)
            struct.pack_into("<H", sec, 0x50, sid & 0xFFFF)
            b.raw = bytes(sec)
    return skel


def _default_split(n: int):
    """3-stream split mirroring native Tigrex's tail-stream sizes (…/9/5)."""
    if n >= 9 + 5 + 1:
        return [n - 14, 9, 5]
    return [n]


def p3rd_to_mhfu(p3rd_blob: bytes, split=None) -> bytes:
    """Convert a MHP3rd (0x80000000) skeleton to a native MHFU 0xC0000000 skeleton.

    The MHP3rd skeleton uses **0x5C** bone sections; the MHFU engine's joint builder
    needs **0x10C** sections (`0x40000001` magic). We rebuild each section from the
    parsed transform fields (idx/parent/child/sibling/scale/rot/pos at the canonical
    offsets — identical across both games) into the full 0x10C layout. The matrix/aux
    region `+0x54..0x10C` is left ZERO (the engine recomputes it at load — verified on
    native `file_06185`). Stream-ids (`+0x50`) are assigned contiguously per ``split``
    so the anim FK partitions cleanly (see :func:`assign_stream_ids`).

    This is the source-skeleton (no-down-rig) path: a ported monster ships its OWN
    skeleton, so its own anim drives it 1:1 (the joint count is data-driven — the engine
    `malloc`s `bone_count*0x250` with no clamp). Returns the 0xC0000000 skeleton bytes.
    """
    from . import skeleton_p3rd as _skp
    ssk = _skp.parse(p3rd_blob)
    bones = ssk.bones
    n = len(bones)
    # preserve the source header's bone_count + animated-count fields
    src_bone_count = struct.unpack_from("<I", p3rd_blob, 4)[0]
    src_animated = struct.unpack_from("<I", p3rd_blob, 0x1C)[0]
    animated = src_animated if 0 < src_animated <= n else n
    sp = split or _default_split(animated)

    runs = []
    for sid, c in enumerate(sp):
        runs += [sid] * c
    while len(runs) < n:
        runs.append(len(sp))                       # non-animated tail -> own stream id

    secs = bytearray()
    for i, b in enumerate(bones):
        sec = bytearray(0x10C)
        struct.pack_into("<I", sec, 0x00, SECTION_MAGIC)
        struct.pack_into("<I", sec, 0x04, b.flag if getattr(b, "flag", 0) else 1)
        struct.pack_into("<I", sec, 0x08, 0x10C)
        struct.pack_into("<4i", sec, 0x0C, b.index, b.parent, b.child, b.sibling)
        struct.pack_into("<3f", sec, 0x1C, *b.bind_scale)
        struct.pack_into("<f",  sec, 0x28, 1.0)
        struct.pack_into("<3f", sec, 0x2C, *b.bind_rot)
        struct.pack_into("<f",  sec, 0x38, 1.0)
        struct.pack_into("<3f", sec, 0x3C, *b.bind_pos)
        struct.pack_into("<f",  sec, 0x48, 1.0)
        struct.pack_into("<i",  sec, 0x4C, -1)
        struct.pack_into("<I",  sec, 0x50, runs[i] & 0xFFFF)
        secs += sec

    total = 0x20 + len(secs)
    hdr = struct.pack("<8I", MAGIC, src_bone_count, total, 0, 2, 0x14, 0, animated)
    return bytes(hdr + secs)


def encode(skel: Skeleton) -> bytes:
    """Serialize a skeleton blob from the data model.

    In-place field patching (the Phase 3 "safe case"): header (verbatim) + per-bone
    sections rebuilt from fields + the opaque bytes after the last parsed section.
    Verified byte-identical for unedited skeletons across all 49 big-monster PACs;
    bind-pose and tree-link edits (same bone count) serialize correctly without
    disturbing the implementation-private matrix/aux region of each section.

    Header words (`bone_count`/`total_size`) are taken verbatim from the source —
    a genuine bone-count change must update the animation pack AND those words in
    lockstep (the validator blocks an unmatched change), so this encoder targets
    the safe same-count case and leaves the header authoritative.
    """
    header = bytearray(skel.header) if skel.header else bytearray(
        struct.pack("<3I", MAGIC, len(skel.bones), 0))
    # If this skeleton came from a MHP3rd PAC (0x80000000 magic), flip the magic to
    # MHFU's 0xC0000000 so the output PAC is recognised by the MHFU engine and by
    # MonsterPac.role().  All other header bytes (bone_count, total_size, extra p3rd
    # word at +0x1C if present) are preserved verbatim.
    if len(header) >= 4 and struct.unpack_from("<I", header, 0)[0] == 0x80000000:
        struct.pack_into("<I", header, 0, MAGIC)
    sections = b"".join(_encode_bone(b) for b in skel.bones)
    body = bytearray(header) + sections
    if skel.raw and len(skel.raw) > len(body):
        body += skel.raw[len(body):]              # preserve the opaque tail
    return bytes(body)
