"""MHFU monster model PAC container (read + byte-exact write).

PAC layout (verified against all big-monster model PACs file_06111..file_06159):
    u32 count
    count × (u32 offset, u32 size)        # INTERLEAVED pairs (NOT two arrays)
    <zero pad to 0x10 boundary>
    sub-resources, each 16-byte aligned, gaps zero-filled
    <opaque tail: bytes after the last sub, up to a 0x800 file boundary>

Empty trailing slots have offset=0, size=0. Sub-resource order in the table
matches ascending file offset. This module reproduces every byte: header pad,
inter-resource zero gaps, and the opaque tail are all preserved, so a
parse→write round-trip is byte-identical, and an edited write (Phase 3) simply
re-flows offsets from the same rules.
"""
from __future__ import annotations

import struct
from dataclasses import dataclass, field
from typing import List, Optional

ALIGN = 0x10


def _align(n: int, a: int = ALIGN) -> int:
    return (n + a - 1) & ~(a - 1)


@dataclass
class SubResource:
    index: int
    data: bytes                 # may be b"" for an empty slot

    @property
    def empty(self) -> bool:
        return len(self.data) == 0

    @property
    def magic(self) -> bytes:
        return self.data[:4]


@dataclass
class MonsterPac:
    subs: List[SubResource] = field(default_factory=list)
    tail: bytes = b""           # opaque bytes after last sub (incl. file padding)
    raw: Optional[bytes] = None  # original bytes, if parsed from a file

    # ---- read ---------------------------------------------------------- #
    @classmethod
    def from_bytes(cls, data: bytes) -> "MonsterPac":
        (count,) = struct.unpack_from("<I", data, 0)
        info = struct.unpack_from("<%dI" % (count * 2), data, 4)
        subs: List[SubResource] = []
        last_end = _align(4 + count * 8)
        for i in range(count):
            off, sz = info[i * 2], info[i * 2 + 1]
            blob = data[off:off + sz] if sz else b""
            subs.append(SubResource(i, blob))
            if sz:
                last_end = max(last_end, off + sz)
        tail = data[last_end:]
        return cls(subs=subs, tail=tail, raw=data)

    # ---- write (byte-exact for unchanged input) ------------------------ #
    def to_bytes(self) -> bytes:
        count = len(self.subs)
        header_end = 4 + count * 8
        pos = _align(header_end)
        table: List[tuple] = [(0, 0)] * count
        placed: List[tuple] = []
        for s in self.subs:
            if s.empty:
                continue
            pos = _align(pos)
            table[s.index] = (pos, len(s.data))
            placed.append((pos, s.data))
            pos += len(s.data)
        body_end = pos
        out = bytearray(body_end)
        struct.pack_into("<I", out, 0, count)
        for i, (off, sz) in enumerate(table):
            struct.pack_into("<II", out, 4 + i * 8, off, sz)
        for off, blob in placed:
            out[off:off + len(blob)] = blob
        out += self.tail
        return bytes(out)

    # ---- convenience --------------------------------------------------- #
    def role(self, sub: SubResource) -> str:
        m = sub.magic
        if m[:4] in (b"\x00\x00\x00\xc0",    # 0xC0000000 MHFU skeleton
                     b"\x00\x00\x00\x80"):    # 0x80000000 MHP3rd skeleton
            return "skeleton"
        if m == b"pmo\x00":
            return "model"
        if m == b".TMH":
            return "texture"
        if sub.data[:1] == b"\x64" and not sub.empty:
            return "anim"
        # MHP3rd lobby anim: first word is species/anim index (small int, e.g. 0x21),
        # total_size == len(sub.data), followed by slot_count and a slot offset table.
        # Detect by: small magic (<= 0xFF), total_size == blob size, and a plausible
        # slot_count (> 0, slot table fits in blob).
        if not sub.empty and len(sub.data) >= 12:
            import struct as _struct
            magic_u32, total_sz, slot_count = _struct.unpack_from("<3I", sub.data, 0)
            if (magic_u32 <= 0xFF and total_sz == len(sub.data) and
                    0 < slot_count <= 512 and 0x0c + slot_count * 4 <= len(sub.data)):
                return "anim"
        return "empty" if sub.empty else "unknown"

    def find(self, role: str) -> Optional[SubResource]:
        for s in self.subs:
            if self.role(s) == role:
                return s
        return None
