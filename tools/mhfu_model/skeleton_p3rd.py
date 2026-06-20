"""MHP3rd skeleton (0x80000000 blob) -> data model.

The MHP3rd skeleton blob is the same family as the MHFU 0xC0000000 blob: same
per-bone section layout (0x10C bytes, magic 0x40000001, fields at identical
offsets), only the top-level magic word differs.  This parser accepts both so
that a caller can use it as a drop-in for the existing MHFU path when the
detected magic is 0x80000000.

The resulting Skeleton dataclass is identical to the one produced by
mhfu_model.skeleton.parse() so the Blender armature builder (_build_armature)
and the exporter (_apply_bindpose) work without modification.

When asset lands a real pmo_p3rd.py / skeleton.parse_p3rd(), that function
should call this one (or inline the same logic) and this file can be retired.
"""
from __future__ import annotations

import struct

from .model import Bone, Skeleton

MAGIC_P3RD = 0x80000000
MAGIC_MHFU = 0xC0000000
SECTION_MAGIC  = 0x40000001   # MHFU + most MHP3rd bones
SECTION_MAGIC2 = 0x40000002   # MHP3rd extra flag (same layout, different flag word)
HDR_SIZE = 0x1C


def parse(blob: bytes) -> Skeleton:
    """Parse a 0x80000000 (MHP3rd) or 0xC0000000 (MHFU) skeleton blob.

    Accepts both magics so callers can funnel either game through here.
    Raises ValueError if the magic is not recognised.
    """
    if len(blob) < HDR_SIZE:
        raise ValueError("skeleton blob too short (%d bytes)" % len(blob))
    magic, bone_count, total_size = struct.unpack_from("<3I", blob, 0)
    if magic not in (MAGIC_P3RD, MAGIC_MHFU):
        raise ValueError("not a recognised skeleton blob (magic=0x%08x)" % magic)

    bones: list[Bone] = []
    # MHP3rd (0x80000000) may carry 1 extra u32 in the header before bone sections,
    # making the effective start 0x20 instead of 0x1C.  Probe both offsets.
    o = HDR_SIZE
    if o + 4 <= len(blob):
        smag_probe = struct.unpack_from("<I", blob, o)[0]
        if smag_probe not in (SECTION_MAGIC, SECTION_MAGIC2) and o + 8 <= len(blob):
            smag_probe2 = struct.unpack_from("<I", blob, o + 4)[0]
            if smag_probe2 in (SECTION_MAGIC, SECTION_MAGIC2):
                o += 4  # skip the extra header word
    hdr_size = o          # 0x1C or 0x20 — the bytes before the first bone section

    for _ in range(bone_count):
        if o + 0x10 > len(blob):
            break
        smag, flag, ssize = struct.unpack_from("<3I", blob, o)
        # Bone index tree and bind-pose at the canonical offsets (same across
        # MHFU and MHP3rd, verified against mhff/psp/pmo.py and iOS m2jean).
        if o + 0x4C > len(blob):
            break
        idx, parent, child, sibling = struct.unpack_from("<4i", blob, o + 0x0C)
        scale = struct.unpack_from("<3f", blob, o + 0x1C)
        rotation = struct.unpack_from("<3f", blob, o + 0x2C)
        position = struct.unpack_from("<3f", blob, o + 0x3C)
        # Clamp section_size to something sane; the section may carry extra
        # IK-chain fields in MHP3rd (fields after 0x48 that MHFU doesn't have).
        sec_sz = ssize if (0 < ssize <= 0x800) else 0x10C
        section = blob[o:o + min(sec_sz, len(blob) - o)]
        bones.append(Bone(
            index=idx, parent=parent, child=child, sibling=sibling,
            bind_scale=scale, bind_rot=rotation, bind_pos=position,
            flag=flag, section_size=sec_sz, raw=section,
        ))
        if smag not in (SECTION_MAGIC, SECTION_MAGIC2) or sec_sz == 0:
            break
        o += sec_sz

    return Skeleton(
        bone_count=bone_count, total_size=total_size, bones=bones,
        header=blob[:hdr_size], raw=blob,
    )
