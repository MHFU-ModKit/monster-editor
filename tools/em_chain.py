#!/usr/bin/env python3
"""Which behaviour pair a handler hands to when it is DONE — the engine's own
sequence, recovered offline from the species overlay.

    python tools/em_chain.py file_06108.bin                # every pair's successors
    python tools/em_chain.py file_06108.bin --pair 1 4     # one pair, every site
    python tools/em_chain.py file_06108.bin --brain        # sites outside handlers

🔴 WHY THIS EXISTS. `em_moveset.py --states` lists what each `(main,sub)` pair DOES;
nothing said what comes AFTER it. A pair's handler does not call `act_set` — none of
em75's 106 do — so the obvious scan finds no hand-offs and the sequence looked like
the brain's secret. It is not: a handler ends by calling the vtable's ENTER-ACTION
slot (`vt+0x88`, `enter(entity, main, id, mode)`), usually through a two-line helper,
and enter-action routes `main` to a per-main TRANSLATOR (`em_moveset.translators`)
that turns the id into the pair. So the successor is a literal in the handler, two
calls away, and this tool reads it back.

    (1,4) charge, phase 3:  collided?           -> enter(0, 6, 1)  -> (0,6)
                            budget +0x76C spent -> helper(1)      -> enter(0, 3, 0) -> (0,3)
                            (+0x280 set)        -> helper         -> enter(2, 2, 0) -> (2,2)

⚠️ That is also why a FORCED pair can last forever. The budget the charge counts down
(`+0x76C`, `0x09AD9A10`) is set by the brain on the way in, not by `act_set`; written
from Lua it has none, and the handler parks in its last phase — 38 s in (1,4), zero
hitbox spawns after the first. The edges here are what the runtime should walk.

The walk is a bounded, path-sensitive interpretation of the handler: literal ints in
registers, the entity pointer, loads from it (`+0x1D5` = the phase), and the return
values of calls, so a branch on any of those becomes a GUARD on the path — "phase==3",
"budget spent", "!collided", "cursor crosses 40". Branches on values it cannot see
fork both ways. Everything it reports is static: a property of the ISO.

What it does NOT see: the brain's own choices (`--brain` lists those sites — their ids
are table-driven, so they read as "computed"), and a hand-off through a call deeper
than two wrappers.
"""
from __future__ import annotations

import argparse
import collections
import json
import struct
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovl_explore import Overlay                                     # noqa: E402
from mips_dis import decode                                         # noqa: E402
import em_moveset as mv                                             # noqa: E402

ENTER_DISPATCH = 0x09AC89F0     # game_task: enter_action(entity, main, id, mode) -> vt+0x88
VT_ENTER = 0x88                 # abi slot 32 — the species' enter-action
ACT_SET = {mv.ACT_SET, mv.ACT_SET_RAW, 0x09AC87E8}
OFF_PHASE = 0x1D5               # entity+0x1D5, the per-action phase byte act_set zeroes

#: predicates a handler branches on, by address. The EBOOT/game_task ones are shared
#: by every species; a species-local one is named where it was read (em75) and shows
#: as hex elsewhere until someone reads it.
PREDICATES = {
    0x08864348: "cursor crosses {f}",     # (clip block, slot, F): 1 the step it crosses F
    0x08864408: "cursor reached {f}",     # (clip block, slot, F): reached-form
    0x09AD9A10: "budget spent",           # +0x76C -= this frame's delta; 1 at <= 0
    0x09D262C0: "collided",               # em75: (+0x278 & 0xC0000003) && +0x27E
    0x08865A9C: "clip done",              # EBOOT anim query (u8 result)
    0x08865A64: "clip query 0x08865A64",
    0x09AD9DD0: "target check 0x09AD9DD0",
    0x09AD9DE0: "target check 0x09AD9DE0",
}

REG_ENT = ("ent",)
OFF_SPECIES = 0x1E8             # entity+0x1E8, the species byte: a constant per port
OFF_MAIN, OFF_SUB = 0x298, 0x299
CALLER_SAVED = list(range(1, 16)) + [24, 25]        # at, v0-v1, a0-a3, t0-t7, t8, t9
A0, A1, A2, A3, T0, V0, T9 = 4, 5, 6, 7, 8, 2, 25


def _f32(bits: int) -> float:
    return struct.unpack("<f", struct.pack("<I", bits & 0xFFFFFFFF))[0]


def _fields(word: int):
    return (word >> 21) & 31, (word >> 16) & 31, (word >> 11) & 31


def _describe(v, fpu) -> str | None:
    """A guard's subject, or None when the value is not worth a guard."""
    if isinstance(v, tuple):
        if v[0] == "mem":
            return "phase" if v[1] == OFF_PHASE else "+0x%X" % v[1]
        if v[0] == "ret":
            fn = v[1]
            name = PREDICATES.get(fn)
            if name is None:
                return "0x%08X()" % fn
            return name.format(f=("%g" % v[2]) if v[2] is not None else "?")
    return None


class Site:
    __slots__ = ("va", "kind", "main", "sub", "mode", "via", "guards", "fn")

    def __init__(self, va, kind, main, sub, mode, via, guards, fn):
        self.va, self.kind, self.main, self.sub, self.mode = va, kind, main, sub, mode
        self.via, self.guards, self.fn = via, guards, fn

    def key(self):
        return (self.va, self.main, self.sub, self.mode, tuple(self.via), tuple(self.guards))

    def doc(self) -> dict:
        return dict(site="0x%08X" % self.va, kind=self.kind, main=self.main, id=self.sub,
                    mode=self.mode, via=["0x%08X" % v for v in self.via],
                    guards=list(self.guards))


def _val(env: dict, r: int):
    return 0 if r == 0 else env.get(r)


class Walker:
    """Path-sensitive walk of one function. Bounded: a pc at most once per path,
    `max_paths` paths, `max_steps` instructions per path.

    `cells` is what a load from those entity offsets evaluates to. `+0x1E8` (the
    species: the overlay serves its subspecies too, em75 = 75/76/88, and every
    handler branches on the byte — a port rides ONE) and `+0x298`/`+0x299` (the
    pair itself: 48 of em75's main-3 subs share one handler that reads `+0x299`
    to know which it is, so without the cell every one of them reports the union
    of all 48 hand-offs)."""

    def __init__(self, ov: Overlay, wrappers: set[int], cells: dict | None = None,
                 max_paths: int = 400, max_steps: int = 700):
        self.ov, self.wrappers, self.cells = ov, wrappers, dict(cells or {})
        self.max_paths, self.max_steps = max_paths, max_steps

    def run(self, fn: int, seed: dict | None = None, depth: int = 0, via=(),
            cells: dict | None = None) -> list[Site]:
        ov = self.ov
        out: list[Site] = []
        seen_sites = set()
        env0 = {A0: REG_ENT}
        if seed:
            env0.update(seed)
        known = dict(self.cells)
        if cells:
            known.update(cells)
        self._known = known
        # (pc, env, fpu, guards, visited, steps)
        work = [(fn, env0, {}, (), frozenset(), 0)]
        paths = 0
        while work and paths < self.max_paths:
            pc, env, fpu, guards, visited, steps = work.pop()
            env, fpu = dict(env), dict(fpu)
            while True:
                if not ov.has(pc) or pc in visited or steps > self.max_steps:
                    paths += 1
                    break
                visited = visited | {pc}
                steps += 1
                ins = decode(ov.word(pc), pc)
                rs, rt, rd = _fields(ins.word)
                op = ins.op

                # ---- straight-line data flow ---------------------------------- #
                if op == "addiu" or op == "addi":
                    if rs == 0:
                        env[rt] = ins.imm
                    elif isinstance(env.get(rs), int):
                        env[rt] = env[rs] + ins.imm
                    else:
                        env.pop(rt, None)
                elif op in ("addu", "or"):
                    if rs == 0 and rt == 0:
                        env[rd] = 0
                    elif rt == 0:
                        env[rd] = env.get(rs)
                    elif rs == 0:
                        env[rd] = env.get(rt)
                    else:
                        env.pop(rd, None)
                    if env.get(rd) is None:
                        env.pop(rd, None)
                elif op == "lui":
                    env[rt] = (ins.imm << 16)
                elif op == "ori":
                    env[rt] = (env[rs] | ins.imm) if isinstance(env.get(rs), int) else None
                    if env[rt] is None:
                        env.pop(rt)
                elif op == "andi":
                    v = env.get(rs)
                    if isinstance(v, int):
                        env[rt] = v & ins.imm
                    elif isinstance(v, tuple) and ins.imm in (0xFF, 0xFFFF):
                        env[rt] = v                      # a mask that keeps the tag
                    else:
                        env.pop(rt, None)
                elif op.startswith("vfpu.0x1F") or (op.startswith("vfpu") and False):
                    # SPECIAL3 seb/seh: `seh rd, rt` keeps the tag; ext/ins drop it
                    if (ins.word & 0x3F) == 0x20:
                        env[rd] = env.get(rt)
                        if env.get(rd) is None:
                            env.pop(rd, None)
                    else:
                        env.pop(rt, None)
                elif op in ("lb", "lbu", "lh", "lhu", "lw"):
                    base = env.get(rs)
                    if base == REG_ENT:
                        env[rt] = ("vt",) if (op == "lw" and ins.imm == 0) else ("mem", ins.imm)
                        if ins.imm in known and op != "lw":
                            env[rt] = known[ins.imm]
                    elif base == ("vt",) and op == "lw":
                        env[rt] = ("vtfn", ins.imm)
                    else:
                        env.pop(rt, None)
                elif op == "mtc1":
                    fr = rd
                    v = env.get(rt)
                    fpu[fr] = _f32(v) if isinstance(v, int) else None
                elif op in ("lwc1",):
                    fpu[rt] = None
                elif op in ("sll", "srl", "sra", "sllv", "srlv", "srav", "sltu", "slt",
                            "slti", "sltiu", "xori", "subu", "sub", "and", "xor", "nor",
                            "mfhi", "mflo", "mfc1"):
                    dst = rt if op in ("slti", "sltiu", "xori") else rd
                    env.pop(dst, None)
                elif op in ("sb", "sh", "sw", "swc1", "nop", "mult", "multu", "div", "divu",
                            "mthi", "mtlo", "cache", "sync") or op.startswith(("c.", "add.",
                            "sub.", "mul.", "div.", "mov.", "neg.", "cvt.", "abs.", "sqrt.",
                            "trunc.", "round.", "floor.", "ceil.", "vfpu")):
                    pass

                # ---- calls -------------------------------------------------- #
                if op in ("jal", "jalr"):
                    d = decode(ov.word(pc + 4), pc + 4)          # delay slot first
                    self._exec_simple(d, env, fpu)
                    target = ins.target if op == "jal" else None
                    is_enter = (op == "jalr" and env.get(rs) == ("vtfn", VT_ENTER)) \
                        or target == ENTER_DISPATCH
                    if is_enter or target in ACT_SET:
                        kind = "enter" if is_enter else "act_set"
                        main, sub, mode = (env.get(A1), env.get(A2), env.get(A3))
                        s = Site(pc, kind,
                                 main if isinstance(main, int) else None,
                                 sub if isinstance(sub, int) else None,
                                 mode if isinstance(mode, int) else None,
                                 list(via), guards, fn)
                        if s.key() not in seen_sites:
                            seen_sites.add(s.key())
                            out.append(s)
                    elif target in self.wrappers and depth < 2:
                        seed2 = {r: env[r] for r in (A0, A1, A2, A3) if r in env}
                        for s in self.run(target, seed2, depth + 1, via + (target,), known):
                            s.guards = guards + s.guards
                            s.fn = fn
                            if s.key() not in seen_sites:
                                seen_sites.add(s.key())
                                out.append(s)
                        self._known = known
                    ret = ("ret", target if target is not None else -1,
                           fpu.get(12) if target in PREDICATES else None)
                    for r in CALLER_SAVED:
                        env.pop(r, None)
                    env[V0] = ret
                    fpu.clear()
                    pc += 8
                    continue

                # ---- control flow --------------------------------------------- #
                if op == "jr":
                    paths += 1
                    break
                if op in ("b", "j"):
                    d = decode(ov.word(pc + 4), pc + 4)
                    self._exec_simple(d, env, fpu)
                    pc = ins.target
                    continue
                if ins.is_branch:
                    likely = op.endswith("l") and op not in ("bgezal", "bltzal")
                    cond, subj, detail = self._cond(op, rs, rt, env, fpu)
                    d = decode(ov.word(pc + 4), pc + 4)
                    if cond is not None:
                        if cond or not likely:
                            self._exec_simple(d, env, fpu)
                        pc = ins.target if cond else pc + 8
                        continue
                    # fork: taken first (pushed), fall-through continues here
                    env_t, fpu_t = dict(env), dict(fpu)
                    self._exec_simple(d, env_t, fpu_t)
                    g_t = guards + ((detail[0],) if subj else ())
                    g_f = guards + ((detail[1],) if subj else ())
                    work.append((ins.target, env_t, fpu_t, g_t, visited, steps))
                    if not likely:
                        self._exec_simple(d, env, fpu)
                    guards = g_f
                    pc += 8
                    continue
                pc += 4
        return out

    def _exec_simple(self, d, env, fpu):
        """A delay slot: only the moves that matter for arguments."""
        rs, rt, rd = _fields(d.word)
        if d.op in ("addiu", "addi"):
            if rs == 0:
                env[rt] = d.imm
            elif isinstance(env.get(rs), int):
                env[rt] = env[rs] + d.imm
            else:
                env.pop(rt, None)
        elif d.op in ("addu", "or"):
            if rs == 0 and rt == 0:
                env[rd] = 0
            elif rt == 0 and rs in env:
                env[rd] = env[rs]
            elif rs == 0 and rt in env:
                env[rd] = env[rt]
            else:
                env.pop(rd, None)
        elif d.op == "lui":
            env[rt] = d.imm << 16
        elif d.op in ("lb", "lbu", "lh", "lhu", "lw"):
            base = env.get(rs)
            if base == REG_ENT:
                env[rt] = ("vt",) if (d.op == "lw" and d.imm == 0) else ("mem", d.imm)
                if d.imm in self._known and d.op != "lw":
                    env[rt] = self._known[d.imm]
            elif base == ("vt",) and d.op == "lw":
                env[rt] = ("vtfn", d.imm)
            else:
                env.pop(rt, None)
        elif d.op == "mtc1":
            v = env.get(rt)
            fpu[rd] = _f32(v) if isinstance(v, int) else None
        elif d.op in ("sb", "sh", "sw", "nop", "swc1") or d.op.startswith(("c.", "vfpu")):
            pass
        elif d.op in ("andi",):
            v = env.get(rs)
            if isinstance(v, int):
                env[rt] = v & d.imm
            elif isinstance(v, tuple) and d.imm in (0xFF, 0xFFFF):
                env[rt] = v
            else:
                env.pop(rt, None)
        else:
            env.pop(rd if d.op in ("sll", "srl", "sra", "sltu", "slt", "subu", "and",
                                   "xor", "nor") else rt, None)

    def _cond(self, op, rs, rt, env, fpu):
        """(truth or None, subject-known, (taken-guard, fallthrough-guard))."""
        a, b = _val(env, rs), _val(env, rt)
        base = op.rstrip("l") if op not in ("bgezal", "bltzal") else op
        if base in ("beq", "bne"):
            if isinstance(a, int) and isinstance(b, int):
                return ((a == b) if base == "beq" else (a != b)), False, None
            subj, lit = (a, b) if isinstance(a, tuple) else (b, a)
            name = _describe(subj, fpu)
            if name is None or not isinstance(lit, int):
                return None, False, None
            if isinstance(subj, tuple) and subj[0] == "ret" and lit in (0, 1):
                # a predicate: `beq v0, zero` taken = it returned 0 = "not name"
                yes, no = ("!" + name, name) if lit == 0 else (name, "!" + name)
            else:
                yes, no = "%s==%d" % (name, lit), "%s!=%d" % (name, lit)
            return None, True, ((yes, no) if base == "beq" else (no, yes))
        if base in ("blez", "bgtz", "bltz", "bgez"):
            if isinstance(a, int):
                return {"blez": a <= 0, "bgtz": a > 0, "bltz": a < 0, "bgez": a >= 0}[base], False, None
            name = _describe(a, fpu)
            if name is None:
                return None, False, None
            sym = {"blez": "<=0", "bgtz": ">0", "bltz": "<0", "bgez": ">=0"}[base]
            neg = {"blez": ">0", "bgtz": "<=0", "bltz": ">=0", "bgez": "<0"}[base]
            return None, True, (name + sym, name + neg)
        return None, False, None                      # bc1t/bc1f: FPU compares, unseen


# --------------------------------------------------------------------------- #
def enter_sites_linear(ov: Overlay, fn: int, limit: int = 1200) -> bool:
    """Does `fn` contain an enter-action or act_set call at all? (cheap pre-pass)"""
    a, n, t9 = fn, 0, None
    while ov.has(a) and n < limit:
        ins = decode(ov.word(a), a)
        rs, rt, rd = _fields(ins.word)
        if ins.op == "lw" and rt == T9 and ins.imm == VT_ENTER and rs == T9:
            t9 = "enter"
        elif ins.op == "lw" and rt == T9:
            t9 = "vt" if ins.imm == 0 else None
        if ins.op == "jalr" and rs == T9 and t9 == "enter":
            return True
        if ins.op == "jal" and (ins.target == ENTER_DISPATCH or ins.target in ACT_SET):
            return True
        if ins.op == "jr" and ins.args == "ra":
            return False
        a += 4
        n += 1
    return False


def wrappers_of(ov: Overlay) -> set[int]:
    """Every overlay function that calls enter-action (directly or via act_set)."""
    out = set()
    t9 = None
    for a in range(ov.text_va, ov.text_end, 4):
        ins = decode(ov.word(a), a)
        rs, rt, rd = _fields(ins.word)
        if ins.op == "lw" and rt == T9:
            t9 = "enter" if (rs == T9 and ins.imm == VT_ENTER) else ("vt" if ins.imm == 0 else None)
        elif ins.op == "jalr" and rs == T9 and t9 == "enter":
            out.add(ov.func_start(a))
        elif ins.op == "jal" and (ins.target == ENTER_DISPATCH or ins.target in ACT_SET):
            out.add(ov.func_start(a))
    return out


# --------------------------------------------------------------------------- #
# id -> pair, through the per-main translators
# --------------------------------------------------------------------------- #
def translator_table(ov: Overlay) -> dict[int, dict[int, list[tuple[int, int]]]]:
    """`{main: {id: [(main', sub), ...]}}` for every main whose enter-action case
    calls a translator with an id switch. Main 0's translator has no switch: it is
    `act_set(0, id)` with one remap (id 1 -> sub 2 under a flag), read by hand."""
    entry, disp = mv.state_dispatchers(ov)
    out: dict[int, dict[int, list[tuple[int, int]]]] = {}
    for fn, sites in mv.translators(ov).items():
        tbl, n, bias = mv.switch_of(ov, fn)
        if not tbl:
            continue
        cases = ov.jumptable(tbl, n or 512)
        idr, _ = mv.param_regs(ov, fn)
        rows: dict[int, list[tuple[int, int]]] = {}
        main_of_fn = None
        for i, ce in enumerate(cases):
            seed = {idr: i + bias} if idr is not None else {}
            r = mv.const_args_at(ov, ce, {mv.ACT_SET, mv.ACT_SET_RAW}, seed=seed)
            if r is None:
                rows[i + bias] = []
                continue
            v, _ = r
            mains = [v["a1"]] if v["a1"] is not None else []
            mains += v.get("a1_alt", [])
            sub = v["a2"] if v["a2"] is not None else i + bias
            rows[i + bias] = [(m, sub) for m in mains]
            if v["a1"] is not None:
                main_of_fn = v["a1"] if main_of_fn is None else main_of_fn
        # which enter-action case routes to this translator = its main
        for m, case_fn in _enter_cases(ov).items():
            if case_fn == fn:
                out[m] = rows
    return out


def _enter_cases(ov: Overlay) -> dict[int, int]:
    """enter-action's `switch(main)`: main -> the translator it jal's, if any.

    ⚠️ Not `em_moveset.switch_of`: enter-action's FIRST `sltiu` is a mode range
    check (`(mode-3) < 2`), which that finder takes for the switch bound. The
    switch is the `jr` through a data-VA table; its bound is the `sltiu` nearest
    above it."""
    out = {}
    enter = _enter_action(ov)
    if enter is None:
        return out
    tbl = n = None
    hi = {}
    for i in range(400):
        a = enter + i * 4
        if not ov.has(a):
            break
        ins = decode(ov.word(a), a)
        rs, rt, rd = _fields(ins.word)
        if ins.op == "lui":
            hi[rt] = ins.imm
        elif ins.op == "addiu" and rs in hi:
            cand = ((hi[rs] << 16) + ins.imm) & 0xFFFFFFFF
            if ov.data_va <= cand < ov.data_end:
                tbl = cand
        elif ins.op == "sltiu":
            n = ins.imm
        elif ins.op == "jr" and ins.args != "ra" and tbl:
            break
        elif ins.op == "jr":
            return out
    if not tbl:
        return out
    for i, ce in enumerate(ov.jumptable(tbl, n or 8)):
        a = ce
        for _ in range(24):
            ins = decode(ov.word(a), a)
            if ins.op == "jal":
                out[i] = ins.target
                break
            if ins.op in ("b", "j", "jr"):
                break
            a += 4
    return out


def _enter_action(ov: Overlay) -> int | None:
    """The species' enter-action: the one function that jal's two or more of the
    id-switch translators (it also calls act_set itself for mains 5 and 7, so it
    is IN `translators()` — that is not a way to exclude it)."""
    switched = {fn for fn in mv.translators(ov) if mv.switch_of(ov, fn)[0]}
    callers: dict[int, set[int]] = {}
    for a in range(ov.text_va, ov.text_end, 4):
        ins = decode(ov.word(a), a)
        if ins.op == "jal" and ins.target in switched:
            fn = ov.func_start(a)
            if fn != ins.target:
                callers.setdefault(fn, set()).add(ins.target)
    best = [fn for fn, t in callers.items() if len(t) >= 2]
    return min(best) if best else None


def resolve(main, sub, table) -> list[tuple[int, int]]:
    """enter(main, id) -> the pair(s) act_set will see."""
    if main is None or sub is None:
        return []
    if main == 0:
        return [(0, sub)] + ([(0, 2)] if sub == 1 else [])
    rows = table.get(main)
    if rows is None:
        return [(main, sub)]
    return rows.get(sub, [(main, sub)]) or []


# --------------------------------------------------------------------------- #
def handlers_of(ov: Overlay) -> dict[tuple[int, int], tuple[int, dict]]:
    """`{(main, sub): (handler, args)}` — `args` are the literal a1/a2/a3/t0 the
    dispatcher's case block passes (main 3's shared handler takes `t0` = variant)."""
    entry, disp = mv.state_dispatchers(ov)
    out = {}
    if entry is None:
        return out
    for m in sorted(disp):
        tbl, n, bias = mv.switch_of(ov, disp[m])
        if not tbl:
            continue
        for sub, ce in enumerate(ov.jumptable(tbl, n or 512)):
            args = {}
            for k in range(5):
                ins = decode(ov.word(ce + k * 4), ce + k * 4)
                rs, rt, rd = _fields(ins.word)
                if ins.op == "addiu" and rs == 0 and rt in (A1, A2, A3, T0):
                    args[rt] = ins.imm
                elif ins.op in ("addu", "or") and rs == 0 and rt == 0 and rd in (A1, A2, A3, T0):
                    args[rd] = 0
                if ins.op == "jal":
                    d = decode(ov.word(ce + k * 4 + 4), ce + k * 4 + 4)
                    drs, drt, drd = _fields(d.word)
                    if d.op == "addiu" and drs == 0 and drt in (A1, A2, A3, T0):
                        args[drt] = d.imm
                    elif d.op in ("addu", "or") and drs == 0 and drt == 0 and drd in (A1, A2, A3, T0):
                        args[drd] = 0
                    out[(m, sub + bias)] = (ins.target, args)
                    break
    return out


def simplify(guards) -> list[str]:
    """One path's guards, readable: a subject that was pinned with `==K` loses its
    earlier `!=` tests (the phase switch is `!=3, !=2, ==1`), duplicates go. A bare
    `+0x280==0` cell test stays: it is what separates (0,3) from (2,2)."""
    pinned = {g.split("==")[0] for g in guards if "==" in g}
    out = []
    for g in guards:
        if "!=" in g and g.split("!=")[0] in pinned:
            continue
        if g not in out:
            out.append(g)
    return out


def _edges(sites: list[Site], table) -> list[dict]:
    """Sites -> edges, one per (site, target set), the alternative guard sets kept.
    `guards` is the SHORTEST alternative — the plainest reason — and `alts` the rest."""
    by_key: dict[tuple, dict] = {}
    for s in sites:
        if s.kind == "enter":
            tgt = resolve(s.main, s.sub, table)
        else:
            tgt = [(s.main, s.sub)] if (s.main is not None and s.sub is not None) else []
        g = simplify(s.guards)
        key = (s.va, tuple(tgt))
        e = by_key.get(key)
        if e is None:
            e = dict(s.doc(), to=[list(p) for p in tgt], computed=not tgt, guards=g,
                     alts=[])
            by_key[key] = e
        else:
            if g != e["guards"] and g not in e["alts"]:
                e["alts"].append(g)
            if len(g) < len(e["guards"]):
                e["alts"].append(e["guards"])
                e["guards"] = g
    return list(by_key.values())


def chain(ov: Overlay, species: int | None = None) -> dict:
    """The whole graph: per pair, its successor edges; plus the brain's sites."""
    if species is None:
        species = mv_species(ov)
    wrappers = wrappers_of(ov)
    table = translator_table(ov)
    trans = set(mv.translators(ov))
    enter = _enter_action(ov)
    handlers = handlers_of(ov)
    walker = Walker(ov, wrappers - trans - ({enter} if enter else set()),
                    {OFF_SPECIES: species} if species is not None else {})
    hands_off = {fn for fn, _ in handlers.values()
                 if fn in wrappers or any(t in wrappers for t in _callees(ov, fn))}
    pairs = {}
    for key, (fn, args) in sorted(handlers.items()):
        sites = walker.run(fn, args, cells={OFF_MAIN: key[0], OFF_SUB: key[1]}) \
            if fn in hands_off else []
        pairs[key] = dict(handler="0x%08X" % fn, args={str(k): v for k, v in args.items()},
                          next=_edges(sites, table))
    # the brain: enter-action callers that are not handlers and not translators
    brain = {}
    hset = {fn for fn, _ in handlers.values()}
    for fn in sorted(wrappers - trans - hset - ({enter} if enter else set())):
        brain["0x%08X" % fn] = _edges(walker.run(fn), table)
    return dict(pairs=pairs, brain=brain, species=species, translators={
        str(m): {str(i): [list(p) for p in v] for i, v in rows.items()}
        for m, rows in table.items()}, enter_action=("0x%08X" % enter) if enter else None)


def mv_species(ov: Overlay) -> int | None:
    name = ov.name.split(".")[0]
    return int(name[2:]) if name.startswith("em") and name[2:].isdigit() else None


def _callees(ov: Overlay, fn: int, limit: int = 1200):
    a, n = fn, 0
    while ov.has(a) and n < limit:
        ins = decode(ov.word(a), a)
        if ins.op == "jal":
            yield ins.target
        if ins.op == "jr" and ins.args == "ra":
            return
        a += 4
        n += 1


# --------------------------------------------------------------------------- #
def fmt_edge(e: dict) -> str:
    to = " ".join("(%d,%d)" % tuple(p) for p in e["to"]) or "(computed)"
    via = " via " + ",".join(e["via"]) if e["via"] else ""
    g = "  [" + " & ".join(e["guards"]) + "]" if e["guards"] else ""
    if e.get("alts"):
        g += "  (or " + " / ".join(" & ".join(a) for a in e["alts"]) + ")"
    what = ("enter(%s,%s,%s)" % (e["main"], e["id"], e["mode"]) if e["kind"] == "enter"
            else "act_set(%s,%s,%s)" % (e["main"], e["id"], e["mode"]))
    return "%s -> %s%s%s   @%s" % (what, to, via, g, e["site"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("overlay")
    ap.add_argument("--pair", nargs=2, type=int, metavar=("MAIN", "SUB"))
    ap.add_argument("--brain", action="store_true", help="sites outside the handlers")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--species", type=int, help="what +0x1E8 reads (default: the overlay's own)")
    args = ap.parse_args()
    ov = Overlay.load_file(args.overlay)
    doc = chain(ov, args.species)
    if args.json:
        print(json.dumps({"%d,%d" % k: v for k, v in doc["pairs"].items()}
                         | {"brain": doc["brain"], "translators": doc["translators"]},
                         indent=1))
        return 0
    if args.brain:
        for fn, sites in doc["brain"].items():
            print("%s: %d site(s)" % (fn, len(sites)))
            for e in sites:
                print("    " + fmt_edge(e))
        return 0
    if args.pair:
        key = tuple(args.pair)
        p = doc["pairs"].get(key)
        if p is None:
            print("no handler for (%d,%d)" % key)
            return 1
        print("(%d,%d) handler %s: %d edge(s)" % (key[0], key[1], p["handler"], len(p["next"])))
        for e in p["next"]:
            print("    " + fmt_edge(e))
        return 0
    n_pairs = len(doc["pairs"])
    with_edges = sum(1 for p in doc["pairs"].values() if p["next"])
    resolved = sum(1 for p in doc["pairs"].values() if any(e["to"] for e in p["next"]))
    succ = collections.Counter()
    for p in doc["pairs"].values():
        for e in p["next"]:
            for t in e["to"]:
                succ[tuple(t)] += 1
    print("%s: %d pair(s) with a handler, %d hand off somewhere, %d to a resolved pair; "
          "enter-action %s" % (ov.name, n_pairs, with_edges, resolved, doc["enter_action"]))
    print("most-entered successors: " + ", ".join(
        "(%d,%d)x%d" % (k[0], k[1], n) for k, n in succ.most_common(8)))
    for key, p in sorted(doc["pairs"].items()):
        if not p["next"]:
            continue
        tos = sorted({tuple(t) for e in p["next"] for t in e["to"]})
        comp = sum(1 for e in p["next"] if not e["to"])
        print("  (%d,%3d) -> %s%s" % (key[0], key[1],
                                      " ".join("(%d,%d)" % t for t in tos) or "-",
                                      "  +%d computed" % comp if comp else ""))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
