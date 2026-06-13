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
    sections = b"".join(_encode_bone(b) for b in skel.bones)
    body = bytearray(header) + sections
    if skel.raw and len(skel.raw) > len(body):
        body += skel.raw[len(body):]              # preserve the opaque tail
    return bytes(body)
