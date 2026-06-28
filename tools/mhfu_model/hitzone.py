"""Monster hitzone / collision data — parse + build + inject.

The per-monster hit/hurt data lives in the host AI overlay (`file_06108` = em75.ovl),
NOT the model PAC. Two tables, referenced by VA pointers in the overlay header (the
injection seam). RE 2026-06-28 (live off the Brute + offline file decode) →
docs/agent_memory_map.md "Hitzone / collision data".

  VOLUMES  (collision spheres, bone-attached — the hurtbox geometry)
    table @ file 0x449C8 (live VA 0x09D5EB48), header ptr @ file 0x46600.
    record 0x28: { bone u32 @+0x00, radius f32 @+0x0C, rest = bone-relative
                   offset/flags (0 on Tigrex) }.  Native Tigrex = 4 recs:
                   bone 10 r150, 18 r150, 41 r170, 42 r170.

  WEAKNESS (per-part damage multipliers)
    records @ file 0x466E0 (live VA 0x09D60848 +0x18 lead-in), header ptr @ 0x470D0.
    record 0x18: { w0/w1 packed element values, part_id u32 @+0x08, flag @+0x0C,
                   type=5 @+0x10, value u32 @+0x14 }.  Native Tigrex = 6 recs.

Injection: overwrite the tables in-place (same record count) or relocate a larger
table and repoint the header VA ptr. Live applier writes the fixed overlay VAs.
The volume table is BONE-INDEXED → indices must match the injected skeleton, so a
ported monster's hitzones pair with the source-skeleton path.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import List

# ---- discovered layout (em75.ovl / file_06108; load base 0x09D1A180) ------- #
VOL_FILE_OFF = 0x449C8
VOL_VA = 0x09D5EB48
VOL_COUNT = 4
VOL_STRIDE = 0x28

WK_FILE_OFF = 0x466E0          # first record (0x18 lead-in after the table base VA)
WK_VA = 0x09D60848 + 0x18
WK_COUNT = 6
WK_STRIDE = 0x18

HDR_PTR_VOL = 0x46600          # file offset where the volumes-table VA is stored
HDR_PTR_WK = 0x470D0           # file offset where the weakness-table VA is stored


@dataclass
class Volume:
    bone: int
    radius: float
    raw: bytes = b""           # full 0x28 record (reserved bytes preserved on edit)

    def pack(self) -> bytes:
        b = bytearray(self.raw if len(self.raw) == VOL_STRIDE else bytes(VOL_STRIDE))
        struct.pack_into("<I", b, 0x00, self.bone & 0xFFFFFFFF)
        struct.pack_into("<f", b, 0x0C, float(self.radius))
        return bytes(b)


@dataclass
class Weakness:
    part: int
    type: int
    value: int
    w0: int = 0
    w1: int = 0
    flag: int = 1
    raw: bytes = b""

    def pack(self) -> bytes:
        b = bytearray(self.raw if len(self.raw) == WK_STRIDE else bytes(WK_STRIDE))
        struct.pack_into("<6I", b, 0,
                         self.w0, self.w1, self.part & 0xFFFFFFFF,
                         self.flag, self.type, self.value & 0xFFFFFFFF)
        return bytes(b)


# --------------------------------------------------------------------------- #
def parse_volumes(buf: bytes, off: int = VOL_FILE_OFF, n: int = VOL_COUNT) -> List[Volume]:
    out = []
    for i in range(n):
        o = off + i * VOL_STRIDE
        rec = buf[o:o + VOL_STRIDE]
        bone = struct.unpack_from("<I", rec, 0)[0]
        radius = struct.unpack_from("<f", rec, 0x0C)[0]
        out.append(Volume(bone=bone, radius=radius, raw=rec))
    return out


def parse_weakness(buf: bytes, off: int = WK_FILE_OFF, n: int = WK_COUNT) -> List[Weakness]:
    out = []
    for i in range(n):
        o = off + i * WK_STRIDE
        rec = buf[o:o + WK_STRIDE]
        w0, w1, part, flag, typ, val = struct.unpack_from("<6I", rec, 0)
        out.append(Weakness(part=part, type=typ, value=val, w0=w0, w1=w1,
                            flag=flag, raw=rec))
    return out


def build_volume_table(vols: List[Volume]) -> bytes:
    return b"".join(v.pack() for v in vols)


def build_weakness_table(wks: List[Weakness]) -> bytes:
    return b"".join(w.pack() for w in wks)


def overwrite_in_place(overlay: bytes, vols: List[Volume] = None,
                       wks: List[Weakness] = None) -> bytes:
    """Return a copy of the overlay with the hitzone tables overwritten IN PLACE
    (same record count — the safe case). Raises if a table would change size."""
    out = bytearray(overlay)
    if vols is not None:
        if len(vols) != VOL_COUNT:
            raise ValueError(f"volume count {len(vols)} != {VOL_COUNT} (use relocate)")
        out[VOL_FILE_OFF:VOL_FILE_OFF + VOL_COUNT * VOL_STRIDE] = build_volume_table(vols)
    if wks is not None:
        if len(wks) != WK_COUNT:
            raise ValueError(f"weakness count {len(wks)} != {WK_COUNT} (use relocate)")
        out[WK_FILE_OFF:WK_FILE_OFF + WK_COUNT * WK_STRIDE] = build_weakness_table(wks)
    return bytes(out)
