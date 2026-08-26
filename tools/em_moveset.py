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

# 🔴 NOT EXTRACTED — the per-slot applier `0x09AC5520(entity, a1 + 0x3E8 +
# slot*0xC8, speed, mode, slot)` also writes the animation channel, one body-part
# slot at a time, from 41 sites in em75.ovl. Un-biasing those into this table was
# tried on 2026-08-26 and DROPPED a1 46 from (0,5) — an id the live trace shows 29
# times — so it is left out rather than shipped with a regression on a confirmed
# row. Live, the missing ids show up as slot disagreement: (0,8) reads `1/24/1`.
# → docs/AI_SCRIPTING_ENGINE.md §34b
PER_SLOT_APPLIER = 0x09AC5520
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


def switch_of(ov: Overlay, fn: int, limit: int = 400, want_key: int | None = None):
    """The (table, count, bias) of a jump-table switch in `fn`, if any.

    🔴 The table address is recognised by where it LANDS, not by how big the
    `%lo` immediate is. An earlier version required `imm > 0x1000`, which happens
    to hold for em75 (11088) and fails for em02 (1688) — so 10 of the 17 monster
    overlays silently reported "no action tick" when they have exactly the same
    structure. Never filter an address by the size of its low half.

    `want_key`: if given, only accept a switch whose bound check reads that entity
    offset (e.g. 0x298), so the species switch in the same function is not
    mistaken for the action dispatch.
    """
    hi: dict[int, int] = {}
    tbl = n = None
    bias = 0
    key_off = None
    insns = []
    for i in range(limit):
        a = fn + i * 4
        if not ov.has(a):
            break
        ins = decode(ov.word(a), a)
        insns.append(ins)
        rs, rt, rd = (ins.word >> 21) & 31, (ins.word >> 16) & 31, (ins.word >> 11) & 31
        if ins.op == "lui":
            hi[rt] = ins.imm
        elif ins.op == "addiu" and rs in hi and ins.imm is not None:
            cand = ((hi[rs] << 16) + ins.imm) & 0xFFFFFFFF
            if ov.data_va <= cand < ov.data_end:
                tbl = cand
        elif ins.op == "sltiu" and n is None:
            n = ins.imm
            src = rs
            # bias must come from the register the bound check reads, not from
            # "the last negative addiu" — every prologue opens `addiu sp, sp, -N`.
            for q in reversed(insns[:-1]):
                qrt, qrs = (q.word >> 16) & 31, (q.word >> 21) & 31
                if q.op == "addiu" and qrt == src and qrs == src and q.imm < 0:
                    bias = -q.imm
                    break
                if q.op == "andi" and qrt == src:
                    src = qrs
                    continue
                if q.op in ("lbu", "lb", "lhu", "lh", "lw") and qrt == src:
                    key_off = q.imm
                    break
                if q.op in ("addiu", "addu", "or") and qrt == src:
                    break
        elif ins.op == "jr" and ins.args != "ra" and tbl:
            if want_key is not None and key_off != want_key:
                hi, tbl, n, bias, key_off = {}, None, None, 0, None
                continue
            return tbl, n, bias
        elif ins.op == "jr" and ins.args == "ra":
            break
    return None, None, 0


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
        tbl, n, _ = switch_of(ov, fn, limit=600, want_key=0x298)
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


# Ops whose destination register is `rt` (I-type) and `rd` (R-type). Used only to
# answer "did something just clobber a1 with a value we cannot read statically?".
_WRITES_RT = {"addiu", "addi", "ori", "andi", "xori", "lui", "slti", "sltiu",
              "lw", "lh", "lhu", "lb", "lbu", "lwl", "lwr", "mfc1", "lwc1"}
_WRITES_RD = {"addu", "add", "subu", "sub", "or", "and", "xor", "nor", "sll",
              "srl", "sra", "sllv", "srlv", "srav", "slt", "sltu", "movn",
              "movz", "mfhi", "mflo"}


def _clobbers(ins, reg: int) -> bool:
    """True if `ins` writes `reg` with something that is not a literal.

    🔴 Without this the tracker never forgets. em75's (1,3) sets `a1 = 2` for one
    branch and then reaches two more executor calls whose `a1` came from a table
    (`lw a1, 0x18(s2)`); the stale 2 was credited to all three, so the row read
    `anim a1 -> 2` as if fully resolved. Live it plays 2, 7 and 8.
    """
    rt, rd = (ins.word >> 16) & 31, (ins.word >> 11) & 31
    if ins.op == "addiu" and ((ins.word >> 21) & 31) == 0:
        return False                                   # that IS the literal form
    return (ins.op in _WRITES_RT and rt == reg) or (ins.op in _WRITES_RD and rd == reg)


def exec_consts(ov: Overlay, fn: int, depth: int = 0, seen=None, limit: int = 1500):
    """Constant `a1` values a handler hands to the animation executor.

    One level of inlining is followed because most handlers are thin wrappers over
    a shared routine that takes the clip id as a parameter. Values marked
    "(computed)" come from a register — usually a rage/variant selector — and need
    a live sweep to pin down.
    """
    seen = seen if seen is not None else set()
    if fn in seen or depth > 1 or not ov.has(fn):
        return set(), False
    seen.add(fn)
    out, regs = set(), {}
    computed = False     # an executor call was reached with a1 holding a non-literal
    alt = set()          # values reachable only via a branch-likely's annulled slot
    for i in range(limit):
        a = fn + i * 4
        if not ov.has(a):
            break
        ins = decode(ov.word(a), a)
        rs, rt = (ins.word >> 21) & 31, (ins.word >> 16) & 31
        if ins.op == "addiu" and rs == 0:
            regs[rt] = ins.imm
        else:
            for r in (5, 8):                # a1 (the id) and t0 (the slot index)
                if _clobbers(ins, r):
                    regs.pop(r, None)
        # 🔴 A branch-LIKELY (`beql`/`bnel`/...) annuls its delay slot when NOT
        # taken, so the slot is the OTHER arm, not part of the straight path.
        # Walking through it linearly reports one arm as if it were the only one:
        # em75's (0,5) is `a1 = 54 if species == 81 else 46`, and the linear read
        # said just 54 — which a live run then contradicted. Keep both.
        if ins.op in ("beql", "bnel", "blezl", "bgtzl", "bltzl", "bgezl", "bc1tl", "bc1fl"):
            d = decode(ov.word(a + 4), a + 4)
            if d.op == "addiu" and ((d.word >> 21) & 31) == 0 and ((d.word >> 16) & 31) == 5:
                alt.add(d.imm)
        if ins.op == "jal":
            d = decode(ov.word(a + 4), a + 4)          # delay slot runs first
            if d.op == "addiu" and ((d.word >> 21) & 31) == 0:
                regs[(d.word >> 16) & 31] = d.imm
            if ins.target == EXECUTOR:
                if 5 in regs:
                    out.add(regs[5])
                else:
                    computed = True
                out |= alt
                alt = set()
            elif ov.text_va <= ins.target < ov.text_end:
                sub_out, sub_computed = exec_consts(ov, ins.target, depth + 1,
                                                    seen, limit)
                out |= sub_out
                computed = computed or sub_computed
        if ins.op == "jr" and ins.args == "ra":
            break
    return out, computed


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
            a1s, computed = exec_consts(ov, tgt)
            a1s = sorted(a1s)
            shown = ",".join(str(x) for x in a1s) if a1s else "(computed)"
            # A handler can have BOTH literal and table-driven executor calls;
            # reporting only the literals reads as "fully resolved" when it isn't.
            if a1s and computed:
                shown += ",+computed"
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
