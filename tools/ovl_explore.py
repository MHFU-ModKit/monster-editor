#!/usr/bin/env python3
"""Read an MWo3 overlay (em*.ovl / *_task.ovl) offline: disassemble, xref, walk switches.

The big-monster AI lives in a per-species overlay that we can only inspect live at
the cost of a cold boot — and reading its code through the debugger returns PPSSPP's
JIT markers, not the game's instructions (`ppsspp-debugging`). This reads the file
from `workspace/extracted/data_files/` instead, so overlay questions cost nothing.

    tools/ovl_explore.py file_06108.bin map
    tools/ovl_explore.py file_06108.bin func 0x09D34390
    tools/ovl_explore.py file_06108.bin xref 0x09D5A580
    tools/ovl_explore.py file_06108.bin switch 0x09D62890

VAs are the addresses the overlay is *loaded* at, so anything printed here can be
pasted straight into a debugger session or a memory map doc.
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from mips_dis import decode                                        # noqa: E402

HDR = struct.Struct("<4sIIIIIII")


class Overlay:
    def __init__(self, data: bytes, path: str = ""):
        magic, self.oid, self.load, self.text_size, self.data_size, \
            self.bss_size, self.ctor_s, self.ctor_e = HDR.unpack_from(data, 0)
        if magic != b"MWo3":
            raise ValueError(f"not an MWo3 overlay: {magic!r}")
        self.name = data[32:64].split(b"\0")[0].decode(errors="replace")
        self.data = data
        self.path = path
        # ⚠️ Code starts at file 0x80, not 0x40: a 64-byte header plus a 64-byte
        # zero pad. Using 0x40 shifts every disassembled VA by 0x40 and the
        # output still *looks* like plausible MIPS.
        self.text_off = 0x80
        self.text_va = self.load + self.text_off
        self.text_end = self.text_va + self.text_size
        self.data_off = self.text_off + self.text_size
        self.data_va = self.load + self.data_off
        self.data_end = self.data_va + self.data_size
        self.bss_end = self.data_end + self.bss_size
        # The image sits ABOVE the slot base; the gap below it is the overlay's
        # lower BSS, which its own code reaches into (BIG_MONSTER_OVERLAY_RELOCATION §2).
        self.slot_base = 0x09D15100 if self.name.startswith("em") else self.load

    @classmethod
    def load_file(cls, p: str | Path) -> "Overlay":
        p = Path(p)
        if not p.exists():
            for cand in (Path("workspace/extracted/data_files") / p.name, ):
                if cand.exists():
                    p = cand
                    break
        return cls(p.read_bytes(), str(p))

    # -- addressing ------------------------------------------------------- #
    def off(self, va: int) -> int:
        return va - self.load

    def has(self, va: int) -> bool:
        return 0 <= va - self.load < len(self.data)

    def word(self, va: int) -> int:
        return struct.unpack_from("<I", self.data, va - self.load)[0]

    def zone(self, va: int) -> str:
        if self.text_va <= va < self.text_end:
            return "text"
        if self.data_va <= va < self.data_end:
            return "data"
        if self.data_end <= va < self.bss_end:
            return "bss"
        if self.slot_base <= va < self.load:
            return "lobss"
        if 0x08800000 <= va < 0x0C000000:
            return "extern"
        return ""

    def in_footprint(self, va: int) -> bool:
        return self.slot_base <= va < self.bss_end

    # -- code ------------------------------------------------------------- #
    def dis(self, va: int, count: int):
        for i in range(count):
            a = va + i * 4
            if not self.has(a):
                return
            yield decode(self.word(a), a)

    def func(self, va: int, limit: int = 4000):
        """Disassemble to the first `jr ra` (plus its delay slot)."""
        out = []
        for ins in self.dis(va, limit):
            out.append(ins)
            if ins.op == "jr" and ins.args == "ra":
                nxt = decode(self.word(ins.va + 4), ins.va + 4)
                out.append(nxt)
                break
        return out

    def func_start(self, va: int, back: int = 3000) -> int:
        """Walk back to the prologue (`addiu sp, sp, -N`) that owns `va`."""
        a = va
        for _ in range(back):
            ins = decode(self.word(a), a)
            if ins.op == "addiu" and ins.args.startswith("sp, sp, -"):
                return a
            a -= 4
            if not self.has(a):
                break
        return va

    # -- references ------------------------------------------------------- #
    def xrefs(self, target: int):
        """Every reference to `target`: j/jal, lui+lo pairs, and data words.

        The hi/lo scan uses register-write dataflow rather than "last lui per
        register": an `addiu R,R,lo` *consumes* the %hi holder, so later `off(R)`
        are struct offsets and pairing them as %lo invents references that do not
        exist (the same trap `ovl_reloc.py` documents for relocation).
        """
        jumps, pairs, words = [], [], []
        hi = {}                       # reg -> (va, imm) while it still holds a %hi
        end = min(self.text_end, self.load + len(self.data))
        for a in range(self.text_va, end, 4):
            ins = decode(self.word(a), a)
            if ins.is_jump and ins.target == target:
                jumps.append(a)
            if ins.op == "lui":
                rt = (ins.word >> 16) & 31
                hi[rt] = (a, ins.imm)
                continue
            rs = (ins.word >> 21) & 31
            rt = (ins.word >> 16) & 31
            if rs in hi and (ins.op in ("addiu", "ori") or ins.imm is not None
                             and ins.op not in ("lui",)):
                lo = ins.imm or 0
                if ins.op == "ori":
                    full = (hi[rs][1] << 16) | (lo & 0xFFFF)
                else:
                    full = ((hi[rs][1] << 16) + lo) & 0xFFFFFFFF
                if full == target:
                    pairs.append((hi[rs][0], a, str(ins)))
                if ins.op == "addiu" and rt == rs:
                    hi.pop(rs, None)          # %hi consumed -> now a real pointer
            # any write to a register kills its %hi
            if ins.op in ("addiu", "addu", "add", "or", "and", "lw", "lbu", "lhu",
                          "lb", "lh", "sll", "srl", "sra", "move", "ori", "andi",
                          "xori", "slti", "sltiu", "mflo", "mfhi", "subu"):
                if rt in hi and ins.op in ("lw", "lbu", "lhu", "lb", "lh", "addiu",
                                           "ori", "andi", "xori", "slti", "sltiu"):
                    hi.pop(rt, None)
        for o in range(self.data_off, self.data_off + self.data_size, 4):
            if struct.unpack_from("<I", self.data, o)[0] == target:
                words.append(self.load + o)
        return jumps, pairs, words

    def jumptable(self, va: int, maxn: int = 512):
        """Read a `jr`-style jump table: consecutive in-text pointers."""
        out = []
        a = va
        while len(out) < maxn and self.has(a):
            w = self.word(a)
            if not (self.text_va <= w < self.text_end):
                break
            out.append(w)
            a += 4
        return out


def _fmt(ov: Overlay, ins) -> str:
    note = ""
    if ins.target is not None and ov.zone(ins.target) == "text":
        note = ""
    return f"  0x{ins.va:08X}  {ins.word:08X}  {ins}{note}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("overlay")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("map")
    p = sub.add_parser("dis");   p.add_argument("va"); p.add_argument("n", nargs="?", default="32")
    p = sub.add_parser("func");  p.add_argument("va")
    p = sub.add_parser("xref");  p.add_argument("va")
    p = sub.add_parser("switch"); p.add_argument("va"); p.add_argument("n", nargs="?", default="512")
    p = sub.add_parser("hex");   p.add_argument("va"); p.add_argument("n", nargs="?", default="128")
    args = ap.parse_args()

    ov = Overlay.load_file(args.overlay)
    va = int(getattr(args, "va", "0"), 0) if hasattr(args, "va") else 0

    if args.cmd == "map":
        print(f"{ov.name}  id={ov.oid}  file={ov.path}")
        print(f"  slot base  0x{ov.slot_base:08X}  (lower bss 0x{ov.load - ov.slot_base:X} bytes)")
        print(f"  load       0x{ov.load:08X}")
        print(f"  text       0x{ov.text_va:08X}..0x{ov.text_end:08X}  (0x{ov.text_size:X})")
        print(f"  data       0x{ov.data_va:08X}..0x{ov.data_end:08X}  (0x{ov.data_size:X})")
        print(f"  bss        0x{ov.data_end:08X}..0x{ov.bss_end:08X}  (0x{ov.bss_size:X})")
        print(f"  ctors      0x{ov.ctor_s:08X}..0x{ov.ctor_e:08X}")
        return 0

    if args.cmd == "dis":
        for ins in ov.dis(va, int(args.n, 0)):
            print(_fmt(ov, ins))
    elif args.cmd == "func":
        f = ov.func(va)
        print(f"function 0x{va:08X}  ({len(f)} insns)")
        for ins in f:
            print(_fmt(ov, ins))
    elif args.cmd == "xref":
        jumps, pairs, words = ov.xrefs(va)
        print(f"xrefs to 0x{va:08X} ({ov.zone(va) or 'outside'}):")
        for a in jumps:
            print(f"  j/jal   0x{a:08X}   (in function 0x{ov.func_start(a):08X})")
        for hi, lo, txt in pairs:
            print(f"  hi/lo   0x{hi:08X}+0x{lo:08X}  {txt}   (in function 0x{ov.func_start(lo):08X})")
        for a in words:
            print(f"  data    0x{a:08X}")
        if not (jumps or pairs or words):
            print("  (none)")
    elif args.cmd == "switch":
        t = ov.jumptable(va, int(args.n, 0))
        print(f"jump table 0x{va:08X}: {len(t)} entries")
        for i, e in enumerate(t):
            print(f"  [{i:3d}] 0x{e:08X}")
    elif args.cmd == "hex":
        n = int(args.n, 0)
        o = ov.off(va)
        for r in range(0, n, 16):
            row = ov.data[o + r:o + r + 16]
            print(f"  0x{va + r:08X}  {row.hex(' ')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
