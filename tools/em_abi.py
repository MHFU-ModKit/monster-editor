#!/usr/bin/env python3
"""em_abi.py — recover the engine <-> big-monster-overlay interface, offline.

A big monster's AI is a per-species MWo3 overlay (`em*.ovl`). All 17 of them load
at the SAME VA (`0x09D1A180`) and NONE has a static constructor, so the engine
cannot be finding entry points by running overlay init code. It finds them the
C++ way: the EBOOT holds a **per-species vtable** whose slots point into overlay
text, and every call the engine makes into a species is a virtual dispatch
through it.

That vtable is the entire entry surface. This tool recovers it from files only —
no cold boot, no debugger — and diffs the 17 species against each other so the
*mandatory* slots (every species overrides them) separate from the optional ones.
Those mandatory slots are the interface a hand-written overlay would have to
implement.

    tools/em_abi.py inventory     # every MWo3 overlay in the extracted data
    tools/em_abi.py vtables       # the 17 entity vtables, attributed to species
    tools/em_abi.py interface     # slot-by-slot diff across all 17 species
    tools/em_abi.py factory       # emId -> species overlay, via the entity factory
    tools/em_abi.py classes em75  # the classes an overlay installs itself
    tools/em_abi.py callers 29    # engine sites that dispatch through a slot

Layout note: the vptr stored at `object+0` points at the `(offset-to-top,
typeinfo)` PAIR, so virtual function *k* lives at `vptr + 8 + 4k`. Slot numbers
here are *k*; the `vt+0xNN` column is the byte offset a `lw` in the disassembly
actually uses.

→ docs/EM_OVERLAY_ABI.md
"""
from __future__ import annotations

import argparse
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovl_explore import Overlay                                     # noqa: E402
from mips_dis import decode                                         # noqa: E402

DATA = Path("workspace/extracted/data_files")
EBOOT = Path("workspace/extracted/PSP_GAME/SYSDIR/BOOT.BIN")
EBOOT_OFF, EBOOT_VA, EBOOT_SZ = 0x25B4, 0x08804000, 0x1C7A90

# The 17 big-monster species overlays, in file order (= FIRST_ID 6094 onward).
EM_SPECIES = ["em01", "em02", "em07", "em14", "em15", "em17", "em20", "em21",
              "em33", "em40", "em54", "em55", "em58", "em59", "em75", "em82",
              "em83"]
EM_FILE = {n: 6094 + i for i, n in enumerate(EM_SPECIES)}

# Every em*.ovl loads here — that is WHY a fixed-VA interface is possible at all.
EM_LOAD = 0x09D1A180
EM_TEXT = 0x09D1A200
EM_TEXT_MAX = 0x09D51A68          # widest species (em75); union bound for "is an
                                  # overlay pointer" tests

# Foreign code regions, for tagging what an unoverridden slot inherits.
ZONES = [(0x09D1A180, 0x09D51A68, "em-ovl"),
         (0x09C19000, 0x09CD0000, "game_sub"),
         (0x09A5F200, 0x09BB0000, "game_task"),
         (0x08804000, 0x089CBA90, "EBOOT")]

ENTITY_VT_SLOTS = 61              # the big-monster entity class


def zone(va: int) -> str:
    for lo, hi, name in ZONES:
        if lo <= va < hi:
            return name
    return "-" if va == 0 else "?"


def is_code(va: int) -> bool:
    return zone(va) not in ("-", "?")


class Eboot:
    def __init__(self, path: Path = EBOOT):
        self.data = path.read_bytes()
        if self.data[:4] != b"\x7fELF":
            raise ValueError(f"{path} is not the decrypted EBOOT ELF")

    def word(self, va: int) -> int:
        if not (EBOOT_VA <= va < EBOOT_VA + EBOOT_SZ):
            raise ValueError(f"VA 0x{va:08X} outside the loaded segment")
        return struct.unpack_from("<I", self.data, EBOOT_OFF + (va - EBOOT_VA))[0]

    def find_vtables(self, lo=0x089B0000, hi=0x089CB000, min_slots=8):
        """A vtable is a (0,0) pair followed by a run of code pointers.

        Both words of the pair are zero in this binary (no RTTI, single
        inheritance for these classes), which makes the scan unambiguous.
        """
        out, va = [], lo
        while va < hi:
            if self.word(va) == 0 and self.word(va + 4) == 0 and is_code(self.word(va + 8)):
                n = 0
                while is_code(self.word(va + 8 + n * 4)):
                    n += 1
                if n >= min_slots:
                    out.append((va, n))
                    va += 8 + n * 4
                    continue
            va += 4
        return out


def load_overlay(species: str) -> Overlay:
    return Overlay((DATA / f"file_0{EM_FILE[species]}.bin").read_bytes())


def _is_entry(ov: Overlay, va: int) -> bool:
    """Does `va` look like a function START in this overlay's image?

    Two accepted shapes: a stack-frame prologue, or an address immediately
    after a `jr ra` + delay slot. Pointing a species' vtable at the WRONG
    overlay lands mid-instruction-stream, which fails both far more often than
    not — enough separation to attribute a vtable to a species.
    """
    if not (ov.text_va <= va < ov.text_end):
        return False
    try:
        if decode(ov.word(va), va).op == "addiu" and "sp, sp, -" in decode(ov.word(va), va).args:
            return True
        return decode(ov.word(va - 8), va - 8).op == "jr"
    except Exception:
        return False


def attribute(eb: Eboot, overlays: dict[str, Overlay]):
    """Map each 61-slot overlay-referencing vtable to the species that owns it.

    Scored by how many of the vtable's overlay pointers land on a plausible
    function entry in each candidate image. Returns [(vtable, species, score)].
    """
    cands = [(va, n) for va, n in eb.find_vtables()
             if n == ENTITY_VT_SLOTS
             and any(EM_LOAD <= eb.word(va + 8 + i * 4) < EM_TEXT_MAX for i in range(n))]
    rows = []
    for va, n in cands:
        ptrs = [eb.word(va + 8 + i * 4) for i in range(n)]
        ep = [p for p in ptrs if EM_LOAD <= p < EM_TEXT_MAX]
        scores = sorted(((sum(_is_entry(overlays[s], p) for p in ep) / len(ep), s)
                         for s in EM_SPECIES), reverse=True)
        rows.append((va, scores[0][1], scores[0][0], scores[1][0], len(ep)))
    return rows


def cmd_inventory(args):
    hdr = struct.Struct("<4sIIIIIII")
    rows = []
    for p in sorted(DATA.glob("file_*.bin")):
        with open(p, "rb") as f:
            h = f.read(64)
        if len(h) < 64 or h[:4] != b"MWo3":
            continue
        _, oid, load, ts, ds, bs, cs, ce = hdr.unpack_from(h, 0)
        name = h[32:64].split(b"\0")[0].decode(errors="replace")
        if args.em and not name.startswith("em"):
            continue
        rows.append((p.name, oid, load, ts, ds, bs, (ce - cs) // 4, name))
    print(f"{'file':16s} {'id':>4s} {'load':>10s} {'text':>8s} {'data':>7s} "
          f"{'bss':>7s} {'ctors':>5s}  name")
    for r in rows:
        print(f"{r[0]:16s} {r[1]:4d} 0x{r[2]:08X} {r[3]:8d} {r[4]:7d} "
              f"{r[5]:7d} {r[6]:5d}  {r[7]}")
    if args.em:
        loads = {r[2] for r in rows}
        ctors = {r[6] for r in rows}
        print(f"\ndistinct load addresses: {[hex(x) for x in loads]}")
        print(f"distinct ctor counts:    {sorted(ctors)}")
        print("→ one fixed load VA + zero constructors = the interface must be "
              "a fixed vtable, not registration code.")


def cmd_vtables(args):
    eb = Eboot()
    ovs = {s: load_overlay(s) for s in EM_SPECIES}
    rows = attribute(eb, ovs)
    print(f"{'vtable':>10s} {'species':>7s} {'ptrs':>4s} {'score':>6s} {'2nd':>6s}   "
          f"overridden slots")
    for va, sp, best, second, nptr in sorted(rows, key=lambda r: EM_SPECIES.index(r[1])):
        slots = [i for i in range(ENTITY_VT_SLOTS)
                 if EM_LOAD <= eb.word(va + 8 + i * 4) < EM_TEXT_MAX]
        print(f"0x{va:08X} {sp:>7s} {nptr:4d} {best:6.0%} {second:6.0%}   "
              f"{','.join(map(str, slots))}")
    seen = [r[1] for r in rows]
    print(f"\n{len(rows)} vtables -> {len(set(seen))} distinct species "
          f"({'1:1' if len(set(seen)) == len(rows) else 'AMBIGUOUS'})")


def cmd_interface(args):
    eb = Eboot()
    ovs = {s: load_overlay(s) for s in EM_SPECIES}
    owner = {sp: va for va, sp, _, _, _ in attribute(eb, ovs)}
    mand, opt, never = [], [], []
    for slot in range(ENTITY_VT_SLOTS):
        vals = {s: eb.word(owner[s] + 8 + slot * 4) for s in EM_SPECIES}
        who = [s for s in EM_SPECIES if EM_LOAD <= vals[s] < EM_TEXT_MAX]
        base = sorted({v for v in vals.values() if not (EM_LOAD <= v < EM_TEXT_MAX)})
        rec = (slot, 8 + slot * 4, who, base, vals)
        (mand if len(who) == 17 else never if not who else opt).append(rec)
    print(f"MANDATORY — every species overrides these ({len(mand)} slots).")
    print("These are the interface a hand-written overlay must implement.\n")
    print(f"  {'slot':>4s} {'vt+':>5s}   {'em75 impl':>10s}")
    for slot, off, _, _, vals in mand:
        print(f"  {slot:4d} 0x{off:03X}   0x{vals['em75']:08X}")
    print(f"\nOPTIONAL — some species override ({len(opt)} slots); base is inherited.\n")
    print(f"  {'slot':>4s} {'vt+':>5s} {'n':>5s}  {'base impl':>10s}  species")
    for slot, off, who, base, _ in opt:
        b = f"0x{base[0]:08X}" if base else "-"
        print(f"  {slot:4d} 0x{off:03X} {len(who):2d}/17  {b:>10s}  {','.join(who)}")
    print(f"\nNEVER overridden: {len(never)} slots — pure base class.")


def cmd_classes(args):
    """Vtables an overlay installs ITSELF (`sw vt, 0(this)` in a constructor)."""
    ov = load_overlay(args.species)
    eb = Eboot()
    regs, found = {}, {}
    for va in range(ov.text_va, ov.text_end, 4):
        try:
            ins = decode(ov.word(va), va)
        except Exception:
            continue
        p = ins.args.replace(",", " ").split()
        if ins.op == "lui" and len(p) >= 2:
            regs[p[0]] = int(p[1], 0) << 16
        elif ins.op == "addiu" and len(p) >= 3 and p[1] in regs:
            regs[p[0]] = (regs[p[1]] + (int(p[2], 0) & 0xFFFFFFFF)) & 0xFFFFFFFF
        elif ins.op == "sw" and p and p[0] in regs and ins.imm == 0:
            v = regs[p[0]]
            if EBOOT_VA <= v < EBOOT_VA + EBOOT_SZ:
                found.setdefault(v, []).append(va)
        elif ins.op in ("jal", "jr", "j"):
            regs = {}
    print(f"{args.species} installs {len(found)} vtables "
          f"(= C++ classes the overlay defines):\n")
    print(f"  {'vtable':>10s} {'slots':>5s} {'ovl':>4s}   ctor sites")
    for v in sorted(found):
        n = 0
        while is_code(eb.word(v + 8 + n * 4)):
            n += 1
        no = sum(1 for i in range(n) if EM_LOAD <= eb.word(v + 8 + i * 4) < EM_TEXT_MAX)
        print(f"  0x{v:08X} {n:5d} {no:4d}   "
              f"{', '.join(f'0x{a:08X}' for a in found[v][:3])}")


EM_FACTORY = 0x09AB15D8           # big-monster entity factory: f(ctx, emId)
EM_FACTORY_TABLE = 0x09C0CCF8     # its 90-entry jump table, indexed by emId
EM_FACTORY_N = 90


def cmd_factory(args):
    """emId -> species overlay, read off the factory's jump table.

    `0x09AB15D8` bounds-checks `emId < 90`, indexes a jump table in game_task's
    .data and `jr`s into a per-species case block. Each block allocates the
    entity and installs that species' vtable, so walking the block to its
    `sw <vtable>, 0(this)` gives the emId -> overlay mapping directly.

    ⚠️ Case blocks are only ~0x60-0xE0 apart, so the walk MUST stop at the next
    case start. An unbounded walk runs into the following block and silently
    reports a neighbouring species (it claimed Popo/Anteka were big monsters).
    """
    eb = Eboot()
    gt = Overlay((DATA / "file_00070.bin").read_bytes())
    vt2sp = {va: sp for va, sp, _, _, _ in attribute(eb, {s: load_overlay(s) for s in EM_SPECIES})}
    tgts = [gt.word(EM_FACTORY_TABLE + i * 4) for i in range(EM_FACTORY_N)]
    bounds = sorted(set(tgts))

    def case_vtable(start):
        regs, end = {}, next((b for b in bounds if b > start), start + 0x200)
        for a in range(start, end, 4):
            try:
                i = decode(gt.word(a), a)
            except Exception:
                continue
            p = i.args.replace(",", " ").split()
            if i.op == "lui" and len(p) >= 2:
                regs[p[0]] = int(p[1], 0) << 16
            elif i.op == "addiu" and len(p) >= 3 and p[1] in regs:
                regs[p[0]] = (regs[p[1]] + (int(p[2], 0) & 0xFFFFFFFF)) & 0xFFFFFFFF
            elif i.op == "sw" and p and p[0] in regs and i.imm == 0 and regs[p[0]] in vt2sp:
                return vt2sp[regs[p[0]]]
            elif i.op in ("jal", "jr", "j", "b"):
                regs = {}
        return None

    res = {e: case_vtable(t) for e, t in enumerate(tgts)}
    by = {}
    for e, s in res.items():
        if s:
            by.setdefault(s, []).append(e)
    print(f"factory 0x{EM_FACTORY:08X}(ctx, emId), jump table 0x{EM_FACTORY_TABLE:08X} "
          f"({EM_FACTORY_N} entries)")
    print(f"{sum(len(v) for v in by.values())} of {EM_FACTORY_N} emIds route to one of "
          f"the {len(by)} species overlays.\n")
    for s in sorted(by, key=lambda x: int(x[2:])):
        ids = ", ".join(f"0x{e:02X}" for e in sorted(by[s]))
        star = "  (primary emId == em number)" if int(s[2:]) in by[s] else ""
        print(f"  {s:6s} <- emId {ids}{star}")
    if args.check:
        print("\nsanity checks (known species):")
        for e, n in ((0x4B, "Tigrex"), (0x4D, "Giadrome"), (0x46, "Popo"), (0x45, "Anteka")):
            print(f"  emId 0x{e:02X} {n:9s} -> {res[e] or 'not a big-monster class'}")


def cmd_callers(args):
    """Engine sites doing `lw t,off(vptr); jalr t` for a given slot.

    ⚠️ The offset alone does not prove the object is a big monster — plenty of
    unrelated classes have a method at the same offset. Treat high counts as
    contaminated and read the low-count sites in context.
    """
    off = 8 + args.slot * 4
    gt = Overlay((DATA / "file_00070.bin").read_bytes())     # game_task.ovl
    hits = []
    for i in range(gt.text_va, gt.text_end - 0x14, 4):
        try:
            ins = decode(gt.word(i), i)
        except Exception:
            continue
        if ins.op != "lw" or ins.imm != off:
            continue
        reg = ins.args.split(",")[0]
        for j in range(i + 4, i + 0x14, 4):
            try:
                nx = decode(gt.word(j), j)
            except Exception:
                continue
            if nx.op == "jalr" and reg in nx.args:
                hits.append(i)
                break
    print(f"slot {args.slot} (vt+0x{off:02X}): {len(hits)} dispatch sites in game_task.ovl")
    for va in hits[:args.limit]:
        print(f"\n  --- 0x{va:08X} ---")
        for a in range(va - 0x0C, va + 0x0C, 4):
            try:
                print(f"    0x{a:08X}  {decode(gt.word(a), a)}")
            except Exception:
                pass


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("inventory"); p.add_argument("--em", action="store_true")
    p.set_defaults(fn=cmd_inventory)
    sub.add_parser("vtables").set_defaults(fn=cmd_vtables)
    sub.add_parser("interface").set_defaults(fn=cmd_interface)
    p = sub.add_parser("classes"); p.add_argument("species", default="em75", nargs="?")
    p.set_defaults(fn=cmd_classes)
    p = sub.add_parser("factory"); p.add_argument("--check", action="store_true")
    p.set_defaults(fn=cmd_factory)
    p = sub.add_parser("callers"); p.add_argument("slot", type=int)
    p.add_argument("--limit", type=int, default=3); p.set_defaults(fn=cmd_callers)
    args = ap.parse_args()
    return args.fn(args) or 0


if __name__ == "__main__":
    raise SystemExit(main())
