"""eboot.py — the main binary, statically, from the ISO.

`PSP_GAME/SYSDIR/EBOOT.BIN` is `~PSP`-encrypted and needs a PSP or a decrypter.
**`BOOT.BIN`, sitting next to it, is the same module as a plain ELF** — no
encryption, no key, nothing to do but read it. Every EBOOT address in
`docs/agent_memory_map.md` resolves in it directly, because the module is
ET_EXEC and loads exactly where it is linked.

That matters because the alternative was a RAM dump, and a RAM dump of a
running game is JIT-MANGLED: PPSSPP overwrites ~25k words of already-translated
code with `0x68xx` EMUHACK markers, so a static disassembly of one is quietly
wrong. This file is the pre-JIT image, always.

    from mhfu_model import eboot
    eb = eboot.load(data_dir)          # data_dir is workspace/extracted
    eb.word(0x088DDB78)                # -> 0x8F39003C
    eb.find(0x8F39003C)                # every site that calls vtable slot 15
    for i in eb.disasm(0x088DDB50, 0x088DDB90): print(i)

Only PT_LOAD segment 0 carries file bytes; the other 299 program headers are
bss reservations for the overlay regions (`0x09A5F200` game_task,
`0x09C19000` game_sub, `0x09D63000` the stage overlays), which is itself a
useful statement of the memory map.
"""
import os
import struct
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import mips_dis                                                   # noqa: E402

BOOT_REL = "PSP_GAME/SYSDIR/BOOT.BIN"
TEXT_VA = 0x08804000
DATA_VA = 0x0890F8F8


class Eboot:
    def __init__(self, blob: bytes):
        if blob[:4] != b"\x7fELF":
            raise ValueError("not an ELF — EBOOT.BIN is encrypted, use BOOT.BIN")
        self.data = blob
        e_phoff, = struct.unpack_from("<I", blob, 28)
        e_phentsize, e_phnum = struct.unpack_from("<HH", blob, 42)
        self.segments = []
        for i in range(e_phnum):
            ptype, off, va, _pa, filesz, memsz, _fl, _al = \
                struct.unpack_from("<8I", blob, e_phoff + e_phentsize * i)
            if ptype == 1 and filesz:
                self.segments.append((va, off, filesz))
        if not self.segments:
            raise ValueError("no loadable segment with file contents")

    def _off(self, va):
        for base, off, size in self.segments:
            if base <= va < base + size:
                return off + va - base
        return None

    def word(self, va):
        o = self._off(va)
        return None if o is None else struct.unpack_from("<I", self.data, o)[0]

    def read(self, va, n):
        o = self._off(va)
        return None if o is None else self.data[o:o + n]

    def find(self, pattern, mask=0xFFFFFFFF):
        """every 4-aligned VA whose word matches — instruction search."""
        out = []
        for base, off, size in self.segments:
            for i in range(0, size & ~3, 4):
                if struct.unpack_from("<I", self.data, off + i)[0] & mask == pattern:
                    out.append(base + i)
        return out

    def callers(self, target):
        """every `jal target`."""
        return self.find(0x0C000000 | ((target >> 2) & 0x03FFFFFF))

    def vtable_calls(self, slot):
        """every `lw t9, slot*4(t9)` — the second half of a virtual dispatch.

        Cheap and surprisingly sharp: slot 15 has exactly two sites in the
        whole binary.
        """
        return self.find(0x8F390000 | (4 * slot))

    def disasm(self, lo, hi):
        for va in range(lo, hi, 4):
            w = self.word(va)
            if w is None:
                continue
            try:
                yield mips_dis.decode(w, va)
            except Exception:
                yield None


def load(data_dir) -> Eboot:
    """`data_dir` is workspace/extracted (the one holding PSP_GAME)."""
    p = os.path.join(data_dir, BOOT_REL)
    if not os.path.exists(p):               # tolerate being handed data_files/
        p = os.path.join(data_dir, "..", BOOT_REL)
    with open(os.path.normpath(p), "rb") as fh:
        return Eboot(fh.read())
