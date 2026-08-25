#!/usr/bin/env python3
"""Extract a big monster's move table from its overlay, offline.

A big monster's behaviour is driven by TWO independent channels:

  * the ANIMATION channel — `executor 0x09AC5228(entity, a1)` fans one action id
    out to all three body-part slots as `a1 + 0x3E8 + slot*0xC8` (`entity+0x324`);
  * the BEHAVIOUR channel — `act_set(entity, main_state, sub_state, mode)`
    (`0x09AC8690` -> `0x09AC8818`) writes `entity+0x298/+0x299` and resets the
    per-slot cursors. The species overlay then runs `switch(+0x298)` ->
    `switch(+0x299)` into per-action CODE, which is what spawns hitboxes and
    applies effects at specific animation frames.

Forcing only the first channel gives the right pixels with the wrong semantics,
which is exactly the "Brute plays the rock-throw while the hunter covers his ears"
result. This tool recovers the *second* channel: the per-species translator
functions that turn an action id into an `act_set(main, sub, mode)`.

    tools/em_moveset.py file_06108.bin

Output is the empirical move table you can script against. Conditional cases
(e.g. "if HP below X use the enraged variant") legitimately emit more than one
row for one id — the condition is in code, so both are listed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovl_explore import Overlay                                     # noqa: E402
from mips_dis import decode                                         # noqa: E402

ACT_SET = 0x09AC8690          # act_set(entity, main, sub, mode) — game_task.ovl
ACT_SET_RAW = 0x09AC8818      # the inner one that actually writes +0x298/+0x299
EXECUTOR = 0x09AC5228         # the animation channel


ARGS = {5: "a1", 6: "a2", 7: "a3"}


def _sim(ov: Overlay, start: int, stop_at: set, limit: int = 400, seed=None):
    """Walk one case block collecting register constants until it reaches act_set.

    Cases share a common tail: the body sets `s1` (the sub_state) and branches to
    one `jal act_set`. So a straight "what was in a2 just before the call" read
    returns a register, not a value — the constant is upstream, in the case body.
    We follow unconditional branches, fall through conditionals, and remember every
    `addiu R, zero, K` seen on the way.

    `alts` collects EVERY constant a register took anywhere in the block, because a
    case that branches on state (HP, rage) legitimately has more than one sub_state
    and taking only the fall-through would silently report one of them as the truth.
    """
    regs, alts = dict(seed or {}), {}
    a = start
    for _ in range(limit):
        ins = decode(ov.word(a), a)
        rt, rs, rd = (ins.word >> 16) & 31, (ins.word >> 21) & 31, (ins.word >> 11) & 31
        if ins.op == "addiu" and rs == 0:
            regs[rt] = ins.imm
            alts.setdefault(rt, set()).add(ins.imm)
        elif ins.op in ("addu", "or") and rs == 0 and rt == 0:
            regs[rd] = 0
            alts.setdefault(rd, set()).add(0)
        elif ins.op in ("addu", "or") and rt == 0 and rs in regs:
            regs[rd] = regs[rs]
            alts.setdefault(rd, set()).add(regs[rs])
        elif ins.op in ("addu", "or"):
            regs.pop(rd, None)
        elif ins.op in ("lbu", "lhu", "lw", "lb", "lh", "lui", "andi", "sll", "srl"):
            regs.pop(rt if ins.op not in ("sll", "srl") else rd, None)
        if ins.op == "jal" and ins.target in stop_at:
            # the delay slot runs before the call
            d = decode(ov.word(a + 4), a + 4)
            if d.op == "addiu" and ((d.word >> 21) & 31) == 0:
                regs[(d.word >> 16) & 31] = d.imm
                alts.setdefault((d.word >> 16) & 31, set()).add(d.imm)
            elif d.op in ("addu", "or") and ((d.word >> 21) & 31) == 0 and ((d.word >> 16) & 31) == 0:
                regs[(d.word >> 11) & 31] = 0
            elif d.op in ("addu", "or") and ((d.word >> 16) & 31) == 0:
                s = (d.word >> 21) & 31
                if s in regs:
                    regs[(d.word >> 11) & 31] = regs[s]
                    alts.setdefault((d.word >> 11) & 31, set()).add(regs[s])
            return a, regs, alts
        if ins.op in ("b", "j") and ins.target and ov.text_va <= ins.target < ov.text_end:
            a = ins.target
            continue
        if ins.op == "jr":
            return None, regs, alts
        a += 4
    return None, regs, alts


def const_args_at(ov: Overlay, case_entry: int, stop_at: set, seed=None):
    call, regs, alts = _sim(ov, case_entry, stop_at, seed=seed)
    if call is None:
        return None
    out = {}
    for r, name in ARGS.items():
        if r in regs:
            out[name] = regs[r]
            extra = alts.get(r, set()) - {regs[r]}
            if extra:
                out[name + "_alt"] = sorted(extra)
        else:
            out[name] = None
    return out, call


def translators(ov: Overlay):
    """Functions that call act_set — i.e. that turn an id into a behaviour state."""
    out = {}
    for a in range(ov.text_va, ov.text_end, 4):
        ins = decode(ov.word(a), a)
        if ins.op == "jal" and ins.target in (ACT_SET, ACT_SET_RAW):
            out.setdefault(ov.func_start(a), []).append(a)
    return out


def param_regs(ov: Overlay, fn: int, limit: int = 60):
    """Which callee-saved registers the prologue parks `a1` (the id) and `a2` in.

    The shared tail reads `sub` out of one of those, so without this the whole
    column resolves to "reg". The default is `sub == the id you passed in`; a case
    only shows a different number when it deliberately remaps itself.
    """
    idr = moder = None
    for i in range(limit):
        ins = decode(ov.word(fn + i * 4), fn + i * 4)
        rs, rt, rd = (ins.word >> 21) & 31, (ins.word >> 16) & 31, (ins.word >> 11) & 31
        if ins.op in ("addu", "or") and rt == 0 and 16 <= rd <= 23:
            if rs == 5 and idr is None:
                idr = rd
            elif rs == 6 and moder is None:
                moder = rd
        if ins.op == "jr":
            break
    return idr, moder


def switch_of(ov: Overlay, fn: int, limit: int = 400):
    """The (table, count) of the first jump-table switch in `fn`, if any."""
    hi = tbl = n = None
    bias = 0
    insns = []
    for i in range(limit):
        a = fn + i * 4
        ins = decode(ov.word(a), a)
        insns.append(ins)
        if ins.op == "lui":
            hi = ins.imm
        elif ins.op == "sltiu" and n is None:
            n = ins.imm
            # ⚠️ The bias must come from the register the bound check reads, not
            # from "the last negative addiu": every prologue starts with
            # `addiu sp, sp, -N`, which otherwise renumbers the whole table.
            src_reg = (ins.word >> 21) & 31
            for p in reversed(insns[:-1]):
                prt, prs = (p.word >> 16) & 31, (p.word >> 21) & 31
                if p.op == "addiu" and prt == src_reg and prs == src_reg and p.imm < 0:
                    bias = -p.imm
                    break
                if p.op in ("addiu", "addu", "andi", "lbu", "lhu", "lw") and prt == src_reg:
                    if p.op == "andi":
                        src_reg = prs
                        continue
                    break
        elif ins.op == "addiu" and hi is not None and ins.imm and ins.imm > 0x1000:
            t = ((hi << 16) + ins.imm) & 0xFFFFFFFF
            if ov.data_va <= t < ov.data_end:
                tbl = t
        elif ins.op == "jr" and ins.args != "ra" and tbl:
            return tbl, n, bias
        elif ins.op == "jr" and ins.args == "ra":
            break
    return None, None, 0



# --------------------------------------------------------------------------- #
# The behaviour side: (main_state, sub_state) -> handler -> the animations it drives
# --------------------------------------------------------------------------- #
def state_dispatchers(ov: Overlay):
    """Find `switch(+0x298) -> switch(+0x299)`, the per-frame action tick.

    `+0x298`/`+0x299` are written only by act_set, never by a species overlay, so
    this is the read side of the behaviour channel: main_state picks a
    sub-dispatcher, sub_state picks the handler that runs the move.

    Detection is deliberately loose — "a function that loads +0x298 and has a jump
    table". Pinning it to em75's exact `lbu/addiu/sb +0x460` prologue found the tick
    in only 3 of the 17 monster overlays; the other 14 shuffle those instructions.
    """
    best = None
    for a in range(ov.text_va, ov.text_end, 4):
        ins = decode(ov.word(a), a)
        if ins.op != "lbu" or ins.imm != 0x298:
            continue
        fn = ov.func_start(a)
        tbl, n, _ = switch_of(ov, fn, limit=600)
        if not tbl or not n:
            continue
        # prefer the tick whose cases actually fan out into +0x299 switches
        cases = ov.jumptable(tbl, n)
        fanout = 0
        for stub in cases:
            for k in range(4):
                j = decode(ov.word(stub + k * 4), stub + k * 4)
                if j.op == "jal" and ov.text_va <= j.target < ov.text_end:
                    t, m, _ = switch_of(ov, j.target, limit=600)
                    if t and m:
                        fanout += 1
                    break
        if best is None or fanout > best[2]:
            best = (fn, tbl, fanout, n)
    if best is None:
        return None, {}
    fn, tbl, _, n = best
    out = {}
    for m, stub in enumerate(ov.jumptable(tbl, n)):
        for k in range(4):
            ins = decode(ov.word(stub + k * 4), stub + k * 4)
            if ins.op == "jal":
                out[m] = ins.target
                break
    return fn, out


def exec_consts(ov: Overlay, fn: int, depth: int = 0, seen=None, limit: int = 1500):
    """Constant `a1` values a handler hands to the animation executor.

    One level of inlining is followed because most handlers are thin wrappers over
    a shared routine that takes the clip id as a parameter. Values marked
    "(computed)" come from a register — usually a rage/variant selector — and need
    a live sweep to pin down.
    """
    seen = seen if seen is not None else set()
    if fn in seen or depth > 1 or not ov.has(fn):
        return set()
    seen.add(fn)
    out, regs = set(), {}
    for i in range(limit):
        a = fn + i * 4
        if not ov.has(a):
            break
        ins = decode(ov.word(a), a)
        rs, rt = (ins.word >> 21) & 31, (ins.word >> 16) & 31
        if ins.op == "addiu" and rs == 0:
            regs[rt] = ins.imm
        if ins.op == "jal":
            d = decode(ov.word(a + 4), a + 4)          # delay slot runs first
            if d.op == "addiu" and ((d.word >> 21) & 31) == 0:
                regs[(d.word >> 16) & 31] = d.imm
            if ins.target == EXECUTOR:
                if 5 in regs:
                    out.add(regs[5])
            elif ov.text_va <= ins.target < ov.text_end:
                out |= exec_consts(ov, ins.target, depth + 1, seen, limit)
        if ins.op == "jr" and ins.args == "ra":
            break
    return out


def print_states(ov: Overlay):
    entry, disp = state_dispatchers(ov)
    if entry is None:
        # Distinguish "this monster has no behaviour channel" (never true) from
        # "its moveset is small enough that the compiler emitted if/else chains
        # instead of a jump table" (true for ~10 of the 17 overlays).
        loads = sum(1 for a in range(ov.text_va, ov.text_end, 4)
                    if decode(ov.word(a), a).op in ("lb", "lbu")
                    and decode(ov.word(a), a).imm in (0x298, 0x299))
        if loads:
            print(f"{ov.name}: reads +0x298/+0x299 at {loads} sites, but its action tick "
                  "dispatches with if/else chains rather than a jump table — the "
                  "(main, sub) model holds, this extractor only handles the table form.")
        else:
            print(f"{ov.name}: no +0x298/+0x299 action tick found")
        return
    print(f"action tick 0x{entry:08X}: switch(+0x298) over {len(disp)} main states\n")
    for m in sorted(disp):
        fn = disp[m]
        tbl, n, bias = switch_of(ov, fn)
        if not tbl:
            print(f"main {m}: 0x{fn:08X}  (no sub_state switch)")
            continue
        cases = ov.jumptable(tbl, n or 512)
        print(f"main {m}: 0x{fn:08X}  {len(cases)} sub_states")
        for sub, ce in enumerate(cases):
            tgt = None
            for k in range(4):
                ins = decode(ov.word(ce + k * 4), ce + k * 4)
                if ins.op == "jal":
                    tgt = ins.target
                    break
            if tgt is None:
                print(f"    ({m},{sub + bias:3d})  0x{ce:08X}  (inline / no handler call)")
                continue
            a1s = sorted(exec_consts(ov, tgt))
            shown = ",".join(str(x) for x in a1s) if a1s else "(computed)"
            print(f"    ({m},{sub + bias:3d})  handler 0x{tgt:08X}  anim a1 -> {shown}")
        print()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("overlay")
    ap.add_argument("--states", action="store_true",
                    help="dump the behaviour side: (main,sub) -> handler -> animation ids")
    args = ap.parse_args()
    ov = Overlay.load_file(args.overlay)
    if args.states:
        print_states(ov)
        return 0
    tr = translators(ov)
    print(f"{ov.name}: {len(tr)} function(s) call act_set "
          f"({sum(len(v) for v in tr.values())} call sites)\n")

    for fn in sorted(tr):
        tbl, n, bias = switch_of(ov, fn)
        calls = sorted(tr[fn])
        stop = {ACT_SET, ACT_SET_RAW}
        if not tbl:
            print(f"function 0x{fn:08X}: {len(calls)} act_set call(s), no id switch")
            for c in calls:
                r = const_args_at(ov, fn, stop)
                if r:
                    v, _ = r
                    print(f"    0x{c:08X}  act_set(main={v['a1']}, sub={v['a2']}, mode={v['a3']})")
            print()
            continue
        cases = ov.jumptable(tbl, n or 512)
        print(f"function 0x{fn:08X}: switch over {len(cases)} ids, table 0x{tbl:08X}")
        idr, moder = param_regs(ov, fn)
        for i, ce in enumerate(cases):
            seed = {}
            if idr is not None:
                seed[idr] = i + bias             # sub defaults to the id passed in
            r = const_args_at(ov, ce, stop, seed=seed)
            if r is None:
                print(f"    id {i + bias:3d}  0x{ce:08X}  (reaches no act_set)")
                continue
            v, call = r
            f = lambda k: ("reg" if v[k] is None else str(v[k])) + (
                "|" + ",".join(str(x) for x in v[k + "_alt"]) if k + "_alt" in v else "")
            print(f"    id {i + bias:3d}  0x{ce:08X}  act_set(main={f('a1'):>8}, "
                  f"sub={f('a2'):>10}, mode={f('a3'):>6})")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
