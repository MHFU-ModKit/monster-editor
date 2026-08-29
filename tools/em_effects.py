#!/usr/bin/env python3
"""Extract a monster's EFFECT vocabulary from its overlay, offline.

A big monster's visual effects are not data. `AI_SCRIPTING_ENGINE.md` §33 settled
that: the species overlay emits them from inside the per-action handler as
literal arguments, `if (frame_reached(F)) spawn_effect(entity, effect_id, bone)`.
There is no per-action effect table to read, inject or port — so the only way to
learn what effects a species can produce is to read them out of its MIPS.

That is what this does. The chain is

    em<N>.ovl  spawn_effect(entity, effect_id, bone)     <- the literal lives HERE
      -> game_task 0x09ACB3E0(entity, id, species_byte, obj, bone, vec3*, mode)
      -> EBOOT     0x08883B54(effect_mgr, ...)

`0x09ACB3E0` is in game_task, which is resident for every species, so it is the
one fixed address the search can start from: find the overlay-local wrapper that
calls it, then find every call to that wrapper and recover the immediates.

🔴 THE ID SPACE IS ALMOST GLOBAL, NOT PER-SPECIES. `0x09ACB3E0` biases the id
only for seven species (`entity+0x1E8`): +30 for 80, +100 for 13/16/30/35/61/62.
Every other species indexes the same library. So an id read out of em17 means
the same effect when em75 asks for it, which is what makes a cross-species
effect census worth building at all — see `--census`.

    tools/em_effects.py file_06108.bin            # one species
    tools/em_effects.py --census                  # all 17, grouped by effect id

⚠️ The bone index is the SPECIES' OWN bone. Effect 60 "at bone 33" is only where
the lightning leaves the mouth if bone 33 is that species' mouth; on a ported
skeleton the same number points somewhere else entirely.
"""
from __future__ import annotations

import argparse
import collections
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovl_explore import Overlay                                     # noqa: E402
from mips_dis import decode                                         # noqa: E402

# game_task, resident for every species: the species-biased effect spawn and the
# two convenience entries that wrap it. Argument positions were read off the
# disassembly, not guessed — each one ends in the same `jal 0x09ACB3E0`:
#
#   0x09ACB3E0(ent, id, base_species, owner, bone, &pos, mode)   the spawn
#   0x09ACB9C8(ent, frame_lo, frame_hi, id, bone, mode, slot)    FRAME-GATED
#   0x09ACB610(ent, id, bone_base, bone_off)                     positional
#
# 0x09ACB9C8 is the interesting one: it fuses `anim_cursor_reached` and the
# spawn into a single call, so it IS the "at frame F, effect E at bone B"
# primitive — 67 call sites across 9 species, and the shape a scripted moveset
# has to reproduce.
SPAWN_BIASED = 0x09ACB3E0
SPAWN_FRAMED = 0x09ACB9C8
SPAWN_POSN   = 0x09ACB610
# entry -> (arg holding the effect id, arg holding the bone)
DIRECT_ENTRIES = {
    SPAWN_BIASED: ("a1", "t0"),
    SPAWN_FRAMED: ("a3", "t0"),
    SPAWN_POSN:   ("a1", "a2"),      # bone is a2+a3; a2 alone is the base
}
DATA_DIR = Path("workspace/extracted/data_files")

# `0x09ACB3E0` adds these to the id before calling the EBOOT spawn, keyed on the
# species byte at entity+0x1E8. Read straight off the switch at 0x09ACB450.
SPECIES_BIAS = {80: 30, 62: 100, 61: 100, 30: 100, 35: 100, 16: 100, 13: 100}

REGNUM = {n: i for i, n in enumerate(
    "zero at v0 v1 a0 a1 a2 a3 t0 t1 t2 t3 t4 t5 t6 t7 "
    "s0 s1 s2 s3 s4 s5 s6 s7 t8 t9 k0 k1 gp sp fp ra".split())}


def em_overlays() -> list[Path]:
    """The 17 em*.ovl, in file order."""
    out = []
    for p in sorted(DATA_DIR.glob("file_0*.bin")):
        try:
            if p.read_bytes()[:4] != b"MWo3":
                continue
            ov = Overlay.load_file(p)
        except Exception:
            continue
        if ov.name.startswith("em"):
            out.append(p)
    return out


def const_arg(ov: Overlay, call_va: int, reg: str, back: int = 40) -> int | None:
    """Recover the constant in `reg` at a call site, scanning backwards.

    ⚠️ Stops at the first control transfer. A value set before a branch that the
    call site is only sometimes reached through is not a value this call always
    passes, and reporting it would invent effect ids that never fire.
    """
    want = REGNUM[reg]
    # the delay slot belongs to the call: check it first
    order = [call_va + 4] + [call_va - 4 * i for i in range(1, back + 1)]
    for va in order:
        if not ov.has(va):
            continue
        ins = decode(ov.word(va), va)
        w = ins.word
        op, rs, rt = w >> 26, (w >> 21) & 31, (w >> 16) & 31
        imm = w & 0xFFFF
        if op == 0x09 and rs == 0 and rt == want:       # addiu rt, zero, imm
            return imm - 0x10000 if imm & 0x8000 else imm
        if op == 0x0D and rs == 0 and rt == want:       # ori rt, zero, imm
            return imm
        if op == 0 and (w & 63) == 0x21 and ((w >> 11) & 31) == want:
            # addu rd, .., ..  -> a register move: the value is not a literal
            return None
        if op in (0x23, 0x24, 0x25, 0x20, 0x21) and rt == want:     # a load
            return None
        if va != call_va + 4 and (ins.is_branch or ins.is_jump or
                                  (ins.op == "jr")):
            return None
    return None


def wrappers(ov: Overlay) -> dict[int, tuple[str, str]]:
    """Overlay-local 3-arg wrappers around the game_task spawn.

    ⚠️ A function that calls the spawn is not automatically a wrapper. em75 has
    six, and three of them are 400-1300 instruction behaviour handlers that
    happen to emit an effect with computed arguments — reading their callers'
    `a1` would invent effect ids. Only a SHORT function is treated as a
    forwarder whose caller supplies the literals.
    """
    found: dict[int, tuple[str, str]] = {}
    va = ov.text_va
    while va < ov.text_end:
        ins = decode(ov.word(va), va)
        if ins.op == "jal" and ins.target == SPAWN_BIASED:
            fn = ov.func_start(va)
            if len(ov.func(fn, 80)) <= 60:
                found[fn] = ("a1", "a2")      # spawn_effect(entity, id, bone)
        va += 4
    return found


def spawns(ov: Overlay) -> list[dict]:
    """Every effect emission with literal arguments, from all four entries."""
    entries = dict(DIRECT_ENTRIES)
    wr = wrappers(ov)
    entries.update(wr)
    out = []
    va = ov.text_va
    while va < ov.text_end:
        ins = decode(ov.word(va), va)
        if ins.op == "jal" and ins.target in entries:
            id_reg, bone_reg = entries[ins.target]
            out.append(dict(site=va, via=ins.target,
                            eid=const_arg(ov, va, id_reg),
                            bone=const_arg(ov, va, bone_reg),
                            frame=(const_arg(ov, va, "a1")
                                   if ins.target == SPAWN_FRAMED else None),
                            local=(ins.target in wr),
                            fn=ov.func_start(va)))
        va += 4
    return out


def report(path: Path) -> None:
    ov = Overlay.load_file(path)
    sp = spawns(ov)
    wr = sorted(wrappers(ov))
    print(f"{path.name}  ({ov.name})  local wrappers: "
          + (", ".join("0x%08X" % w for w in wr) or "none"))
    named = [s for s in sp if s["eid"] is not None]
    print(f"  {len(sp)} spawn sites, {len(named)} with a literal id, "
          f"{len(sp) - len(named)} computed")
    by_fn = collections.defaultdict(list)
    for s in named:
        by_fn[s["fn"]].append(s)
    for fn in sorted(by_fn):
        rows = by_fn[fn]
        seen = sorted({(r["eid"], r["bone"], r["frame"]) for r in rows})
        print("  handler 0x%08X  %2d sites  " % (fn, len(rows))
              + " ".join("%d@b%s%s" % (i, b if b is not None else "?",
                                       "" if f is None else "@f%d" % f)
                         for i, b, f in seen))


def census(paths: list[Path]) -> None:
    """Which species use which effect id — the cross-species dictionary."""
    use: dict[int, dict[str, int]] = collections.defaultdict(
        lambda: collections.defaultdict(int))
    per: dict[str, set] = {}
    for p in paths:
        ov = Overlay.load_file(p)
        ids = set()
        for s in spawns(ov):
            if s["eid"] is None:
                continue
            use[s["eid"]][ov.name] += 1
            ids.add(s["eid"])
        per[ov.name] = ids
    print("=== per species ===")
    for name in sorted(per):
        ids = sorted(per[name])
        print("%-8s %3d ids: %s" % (name, len(ids),
                                    " ".join(str(i) for i in ids)))
    print()
    print("=== per effect id (shared ids are the portable vocabulary) ===")
    for eid in sorted(use):
        users = use[eid]
        print("%4d  %2d species  %s" % (
            eid, len(users),
            " ".join("%s:%d" % (k, v) for k, v in sorted(users.items()))))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("overlay", nargs="*", help="em*.ovl file(s); default = all")
    ap.add_argument("--census", action="store_true",
                    help="cross-species effect-id table instead of per-species detail")
    a = ap.parse_args()
    paths = [Path(x) if Path(x).exists() else DATA_DIR / x for x in a.overlay]
    if not paths:
        paths = em_overlays()
    if a.census:
        census(paths)
    else:
        for p in paths:
            report(p)
            print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
