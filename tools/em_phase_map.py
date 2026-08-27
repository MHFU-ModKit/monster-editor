#!/usr/bin/env python3
"""em_phase_map.py — what actually decides how long a big-monster action runs.

`entity+0x414` is real but a MINORITY gate — 27 of em75's 231 actions, not the
universal action clock an earlier note here claimed. Most actions read 0 from it
throughout. The dominant timing lives in the per-`(main,sub)` handler, which is a
small phase machine on `entity+0x1D5` whose transitions are gated on the **clip's
own cursor**:

  `0x08864408(block, slot, frame)` -> 1 when the clip cursor has reached `frame`
      lb   v0, 0x3E(a0)          ; disabled? -> 0
      lwc1 f0, 0x10(a0+slot*64)  ; the live cursor (entity+0x80 +slot*0x40 +0x10)
      c.le.s f12, f0             ; threshold <= cursor
  `0x08864348(block, slot, frame)` -> the windowed form (cursor inside a range),
      which is what a hitbox-active test looks like; 280 call sites in em75.
  `entity+0xBC & 1`              -> the clip is still PLAYING; cleared at the end.

That split is the whole answer for a port:

  * an action's **total length is clip-driven** (the last phase waits on `+0xBC`),
    so a ported clip may be any length — it is NOT truncated;
  * but the **intra-clip event frames are hardcoded** (the roar tests 60.0), so a
    ported clip must put its impact on the frame the handler expects, or the
    effect fires at the wrong moment.

This tool extracts those numbers per action.

    tools/em_phase_map.py file_06108.bin                # every (main,sub)
    tools/em_phase_map.py file_06108.bin --main 0       # one main state
    tools/em_phase_map.py file_06108.bin --pair 0,4     # one action, verbose

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
from em_moveset import switch_of, state_dispatchers                 # noqa: E402

CURSOR_REACHED = 0x08864408       # (block, slot, frame) -> cursor >= frame
CURSOR_WINDOW = 0x08864348        # (block, slot, frame) -> windowed form
PHASE = 0x1D5                     # entity+0x1D5, the handler's phase byte
CLIP_FLAGS = 0xBC                 # bit 0 = clip still playing
TIMER = 0x414                     # the countdown that is NOT the duration gate
ANIM_BLOCK = 0x80                 # entity+0x80; per-slot stride 0x40, cursor +0x10


def f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def analyse(ov: Overlay, fn: int, cap: int = 600):
    """Walk one handler; return the gates it uses."""
    gates = {"frames": [], "windows": [], "clip_done": 0, "timer": 0, "phases": set()}
    regs: dict[str, int] = {}
    f12: int | None = None
    for k in range(cap):
        va = fn + k * 4
        try:
            i = decode(ov.word(va), va)
        except Exception:
            break
        p = i.args.replace(",", " ").split()
        if i.op == "lui" and len(p) >= 2:
            try:
                regs[p[0]] = int(p[1], 0) << 16
            except Exception:
                pass
        elif i.op in ("addiu", "ori") and len(p) >= 3 and p[1] in regs:
            try:
                regs[p[0]] = (regs[p[1]] + (int(p[2], 0) & 0xFFFFFFFF)) & 0xFFFFFFFF
            except Exception:
                pass
        elif i.op == "mtc1" and len(p) >= 2 and p[1] == "$f12" or (
                i.op == "mtc1" and len(p) >= 2 and p[1] == "f12"):
            f12 = regs.get(p[0])
        elif i.op == "lwc1" and "f12" in i.args:
            f12 = None                          # loaded from data, not a literal
        elif i.op in ("lbu", "lb") and i.imm == PHASE:
            pass
        elif i.op == "sb" and i.imm == PHASE:
            gates["phases"].add(va)
        elif i.op in ("lhu", "lh", "lw") and i.imm == CLIP_FLAGS:
            gates["clip_done"] += 1
        elif i.op == "lw" and i.imm == TIMER:
            gates["timer"] += 1
        elif i.op == "jal" and i.target in (CURSOR_REACHED, CURSOR_WINDOW):
            key = "frames" if i.target == CURSOR_REACHED else "windows"
            gates[key].append(None if f12 is None else round(f32(f12), 2))
            f12 = None
        if i.op == "jr" and "ra" in i.args:
            break
    return gates


def handlers(ov: Overlay):
    """(main, sub) -> handler address, reusing em_moveset's dispatcher walk."""
    _entry, disp = state_dispatchers(ov)
    out = {}
    for m, fn in sorted(disp.items()):
        tbl, n, bias = switch_of(ov, fn)
        if not tbl:
            continue
        for sub, ce in enumerate(ov.jumptable(tbl, n or 512)):
            tgt = None
            for k in range(4):
                ins = decode(ov.word(ce + k * 4), ce + k * 4)
                if ins.op == "jal":
                    tgt = ins.target
                    break
            if tgt is not None and ov.text_va <= tgt < ov.text_end:
                out[(m, sub + bias)] = tgt
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("overlay")
    ap.add_argument("--main", type=int)
    ap.add_argument("--pair")
    args = ap.parse_args()
    ov = Overlay.load_file(args.overlay)
    hs = handlers(ov)

    if args.pair:
        m, s = (int(x) for x in args.pair.split(","))
        fn = hs.get((m, s))
        if fn is None:
            print(f"({m},{s}) has no handler")
            return 1
        g = analyse(ov, fn)
        print(f"({m},{s})  handler 0x{fn:08X}")
        print(f"  event frames tested : {g['frames']}")
        print(f"  windowed tests      : {g['windows']}")
        print(f"  clip-done (+0xBC)   : {g['clip_done']}")
        print(f"  timer  (+0x414)     : {g['timer']}")
        print(f"  phase writes        : {len(g['phases'])}")
        return 0

    print(f"{ov.name}: {len(hs)} (main,sub) handlers\n")
    print(f"{'pair':>9s} {'handler':>10s} {'frames tested':>34s} {'win':>4s} "
          f"{'clip':>4s} {'timr':>4s}")
    clip_only = frame_only = both = neither = 0
    unknown_n = [0]
    seen = {}
    for (m, s), fn in sorted(hs.items()):
        if args.main is not None and m != args.main:
            continue
        if fn in seen:
            g = seen[fn]
        else:
            g = seen[fn] = analyse(ov, fn)
        fr = ",".join("?" if x is None else f"{x:g}" for x in g["frames"]) or "-"
        print(f"  ({m:2d},{s:3d}) 0x{fn:08X} {fr[:34]:>34s} {len(g['windows']):4d} "
              f"{g['clip_done']:4d} {g['timer']:4d}")
        # Classify by what ENDS the action. clip-done wins when present: the
        # handler cannot advance past that phase until the clip stops playing,
        # whatever else it also tests.
        if g["clip_done"]:
            (both if g["frames"] or g["windows"] else clip_only)
            if g["frames"] or g["windows"]:
                both += 1
            else:
                clip_only += 1
        elif g["timer"]:
            frame_only += 1                    # +0x414 countdown: FIXED length
        elif g["frames"] or g["windows"]:
            neither += 1                       # cursor-gated but never waits for the end
        else:
            unknown_n[0] += 1
    tot = both + clip_only + frame_only + neither + unknown_n[0]
    unresolved = sum(1 for g in seen.values() for x in g["frames"] if x is None)
    print(f"\n  of {tot} actions, by what ENDS them:")
    print(f"    {clip_only:3d} clip-done only        -> any clip length, no event frames")
    print(f"    {both:3d} clip-done + cursor tests -> any clip length, but FIXED event frames")
    print(f"    {frame_only:3d} +0x414 countdown      -> FIXED length, clip is ignored")
    print(f"    {neither:3d} cursor tests only")
    print(f"    {unknown_n[0]:3d} no gate found         (instant / driven from elsewhere)")
    ends_on_clip = both + clip_only
    print(f"\n  {ends_on_clip}/{tot} ({ends_on_clip/tot:.0%}) END WHEN THE CLIP ENDS"
          f" -> their length is the ported clip's to choose.")
    print(f"  {frame_only}/{tot} ({frame_only/tot:.0%}) run a fixed frame count regardless"
          f" -> a longer ported clip IS truncated here.")
    if unresolved:
        print(f"\n  \u26a0 {unresolved} cursor thresholds are loaded from data, not literals,"
              f" and show as '?'.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
