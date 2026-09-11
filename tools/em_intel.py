#!/usr/bin/env python3
"""Join every offline analyser for one species into `species/emNN.json`.

Four tools already answer "what does the host action expect?", each in its own
format and none of them joined:

    em_moveset.py <ovl> --states   every reachable (main,sub) and its a1 clip ids
    em_phase_map.py <ovl>          what ENDS the action: clip-done / cursor
                                   frames / the +0x414 budget (and who owns it)
    em_effects.py <ovl>            per-handler effect recipes (id @ bone @ frame)
    em_chain.py <ovl>              what comes NEXT: the pair(s) a handler hands
                                   to when it ends, and the guard on each edge
    em_state_census.py             MEASURED dwell per pair, and whether it moves

This emits one file keyed by `(main, sub)` that carries all four, and — the point
of the exercise — says **per field** whether the number came from reading the
overlay's MIPS (`static`) or from watching the game (`measured`). The UI renders
them differently because they are not the same kind of fact: static intel is a
property of the ISO and never changes, a measurement is a sample of one session.

    tools/em_intel.py file_06108.bin                  # -> species/em75.json
    tools/em_intel.py --all                           # all 17 em*.ovl
    tools/em_intel.py file_06108.bin --log <framework.log> --census-species 75

🔴 THE CENSUS IS USUALLY ABSENT, AND THAT IS THE NORMAL CASE. Collecting it needs
a cold boot with the observe-only probe deployed; `framework.log` on a fresh
checkout has no `[state]` lines at all. So the file is written anyway, with
`census.present = false`, a `reason`, and every pair's `measured` block `null`.
Nothing is estimated or back-filled. An invented dwell would be worse than an
absent one — the whole value of the census field is that it is the only measured
thing in the file, and it is the only source that knows whether a pair is
*usable*: **411 of 411 forced moves into never-entered pairs survived exactly one
tick** (`monster-ai`).

⚠️ `em_state_census.py` reads `[state] main=.. sub=..` lines that do not record
which species produced them, so a log cannot be attributed automatically. Pass
`--census-species` (or generate a single overlay, which implies it). Attaching a
Tigrex census to em17.json would turn every Rathalos pair into a measured lie.

Effects join on the HANDLER FUNCTION, not on the pair, because that is the only
edge the code actually has: `em_effects` recovers the literal arguments at each
`spawn_effect` site and reports the function they sit in. A site is credited to
`(main,sub)` when its function is reachable from that pair's handler by direct
calls. em75's richest effect routines are NOT reachable that way — they hang off
a species-byte switch (`entity+0x1E8`) that no pair handler calls — so they are
listed once, at the top level, in `unattributed_effects` rather than being
spread across pairs on a guess.

→ docs/AI_SCRIPTING_ENGINE.md §33-34, docs/EM_OVERLAY_ABI.md, docs/EFFECTS_AND_VFX.md
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from ovl_explore import Overlay                                     # noqa: E402
from mips_dis import decode                                         # noqa: E402
import em_effects as fx                                             # noqa: E402
import em_moveset as mvs                                            # noqa: E402
import em_phase_map as pm                                           # noqa: E402
import em_state_census as cs                                        # noqa: E402
import em_attacks as atk                                            # noqa: E402
import em_chain as chn                                              # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent / "mhfu_model"))
import hitzone as hz                                                # noqa: E402
import hitbox as hb                                                 # noqa: E402

SCHEMA = "mhfu.species_intel/1"
OUT_ROOT = Path("species")

#: how far the call graph is followed when crediting an effect site to a pair.
#: `em_moveset.exec_consts` follows one level for the same reason — most handlers
#: are thin wrappers. Two is free here and measurably changes nothing on em75.
EFFECT_CALL_DEPTH = 2

#: game_task entry -> what the call site means, for the `via` field.
VIA = {fx.SPAWN_BIASED: "biased", fx.SPAWN_FRAMED: "framed",
       fx.SPAWN_POSN: "positional"}

#: `PROVENANCE` values. A field is one of exactly these three, never blank.
STATIC = "static"          # read out of the overlay's MIPS; a property of the ISO
MEASURED = "measured"      # observed in a running game; a sample, not a law
ABSENT = "absent"          # no evidence of that kind was supplied


# --------------------------------------------------------------------------- #
# the call graph — needed to credit an effect site to a (main,sub)
# --------------------------------------------------------------------------- #
def call_graph(ov: Overlay) -> tuple[list[int], dict[int, set[int]]]:
    """`(prologues, fn -> set of in-overlay callees)`.

    ⚠️ Functions are bounded by the NEXT prologue, not by the first `jr ra`.
    Half these handlers return early from a phase test, and stopping at the first
    `jr ra` truncates them to a dozen instructions — which reports a handler that
    emits four effects as emitting none.
    """
    pro = []
    for a in range(ov.text_va, ov.text_end, 4):
        ins = decode(ov.word(a), a)
        if ins.op == "addiu" and ins.args.startswith("sp, sp, -"):
            pro.append(a)
    calls: dict[int, set[int]] = {}
    for i, fn in enumerate(pro):
        end = pro[i + 1] if i + 1 < len(pro) else ov.text_end
        out: set[int] = set()
        for a in range(fn, end, 4):
            ins = decode(ov.word(a), a)
            if (ins.op == "jal" and ins.target
                    and ov.text_va <= ins.target < ov.text_end):
                out.add(ins.target)
        calls[fn] = out
    return pro, calls


def reachable(calls: dict[int, set[int]], fn: int, depth: int) -> set[int]:
    seen = {fn}
    frontier = [fn]
    for _ in range(depth):
        nxt = []
        for f in frontier:
            for c in calls.get(f, ()):
                if c not in seen:
                    seen.add(c)
                    nxt.append(c)
        frontier = nxt
    return seen


# --------------------------------------------------------------------------- #
# the static half
# --------------------------------------------------------------------------- #
def ends_on(gates: dict) -> str:
    """What terminates the action — em_phase_map's own classification, named.

    clip-done wins when present: the handler cannot advance past that phase until
    the clip stops playing, whatever else it also tests.
    """
    if gates["clip_done"]:
        return "clip+cursor" if (gates["frames"] or gates["windows"]) else "clip"
    if gates["timer"]:
        return "budget"
    if gates["frames"] or gates["windows"]:
        return "cursor"
    return "unknown"


def _case_handler(ov: Overlay, case_entry: int) -> int | None:
    for k in range(4):
        ins = decode(ov.word(case_entry + k * 4), case_entry + k * 4)
        if ins.op == "jal":
            return ins.target if ov.text_va <= ins.target < ov.text_end else None
    return None


def static_intel(ov: Overlay) -> dict:
    """Everything three offline analysers know about this overlay, joined."""
    entry, disp = mvs.state_dispatchers(ov)
    mains: list[dict] = []
    cases: dict[tuple[int, int], int | None] = {}
    if entry is not None:
        for m in sorted(disp):
            fn = disp[m]
            tbl, n, bias = mvs.switch_of(ov, fn)
            if not tbl:
                # 🔴 NOT "this main state has no actions". The extractor only
                # handles the jump-table form; mains 5/6/7 of em75 dispatch some
                # other way. Saying so is the difference between "no such action"
                # and "we cannot see it".
                mains.append(dict(main=m, dispatcher="0x%08X" % fn,
                                  sub_states=None, enumerated=False,
                                  note="no sub_state jump table — this main "
                                       "state's actions are not enumerable "
                                       "offline"))
                continue
            tbl_cases = ov.jumptable(tbl, n or 512)
            mains.append(dict(main=m, dispatcher="0x%08X" % fn,
                              sub_states=len(tbl_cases), enumerated=True,
                              first_sub=bias, note=""))
            for sub, ce in enumerate(tbl_cases):
                cases[(m, sub + bias)] = _case_handler(ov, ce)

    # effects, by the function they sit in
    sites = fx.spawns(ov)
    by_fn: dict[int, list[dict]] = collections.defaultdict(list)
    for s in sites:
        by_fn[s["fn"]].append(s)
    # attack spawns, the same way (#33): the literal id each handler hands the
    # species' node constructor, credited through the same call graph
    atk_sites = atk.sites(ov)
    atk_by_fn: dict[int, list[dict]] = collections.defaultdict(list)
    for s in atk_sites:
        atk_by_fn[s["fn"]].append(s)
    atk_credited: set[int] = set()
    _pro, calls = call_graph(ov)

    gate_cache: dict[int, dict] = {}
    a1_cache: dict[int, tuple[list[int], bool]] = {}
    seed_cache: dict[int, list[int]] = {}
    credited: set[int] = set()          # spawn site VAs credited to some pair

    pairs: dict[tuple[int, int], dict] = {}
    for (m, sub), h in sorted(cases.items()):
        rec: dict = dict(main=m, sub=sub,
                         handler=None if h is None else "0x%08X" % h)
        if h is None:
            rec["note"] = ("the dispatcher's case for this pair runs inline and "
                           "calls no handler — nothing offline can say what it does")
            pairs[(m, sub)] = rec
            continue
        if h not in gate_cache:
            gate_cache[h] = pm.analyse(ov, h)
            a1s, computed = mvs.exec_consts(ov, h)
            a1_cache[h] = (sorted(a1s), computed)
        g = gate_cache[h]
        a1s, a1_computed = a1_cache[h]
        end = ends_on(g)
        rec["a1"] = a1s
        rec["a1_computed"] = a1_computed
        rec["ends_on"] = end
        rec["event_frames"] = [None if x is None else round(x, 2)
                               for x in g["frames"]]
        rec["windows"] = len(g["windows"])
        # 🔴 the windowed test's LITERALS, not just how many there were. `0x08864348`
        # is the shape of a hitbox-active check (280 call sites in em75), so these are
        # the frames a ported clip has to put its impact between — exactly the numbers
        # the action inspector (#9) draws on the timeline. Keeping only the count made
        # the richest gate in the overlay unusable.
        rec["window_frames"] = [None if x is None else round(x, 2)
                                for x in g["windows"]]
        rec["clip_done_reads"] = g["clip_done"]
        rec["budget_reads"] = g["timer"]
        if end == "budget":
            if h not in seed_cache:
                seed_cache[h] = pm.budget_owner(ov, h)
            seeds = seed_cache[h]
            rec["budget"] = dict(gated=True, phase0_seeds=seeds,
                                 post_hook_owns=not seeds)
        else:
            rec["budget"] = dict(gated=False, phase0_seeds=[],
                                 post_hook_owns=None)
        eff = []
        for f in reachable(calls, h, EFFECT_CALL_DEPTH):
            for s in by_fn.get(f, ()):
                if s["eid"] is None:
                    continue
                credited.add(s["site"])
                eff.append(dict(id=s["eid"], bone=s["bone"], frame=s["frame"],
                                site="0x%08X" % s["site"],
                                via="local" if s["local"] else VIA.get(s["via"], "?"),
                                fn="0x%08X" % s["fn"]))
        eff.sort(key=lambda e: (e["id"], e["bone"] if e["bone"] is not None else -1,
                                e["frame"] if e["frame"] is not None else -1))
        rec["effects"] = eff
        # the attacks this pair's handler can spawn — HANDLER literals, before any
        # species id offset (`hitbox.id_offset`). Sorted unique; a computed id is
        # counted, not invented.
        aids: set[int] = set()
        a_sites = 0
        a_computed = 0
        for f in reachable(calls, h, EFFECT_CALL_DEPTH):
            for s in atk_by_fn.get(f, ()):
                a_sites += 1
                atk_credited.add(s["site"])
                if s["aid"] is None:
                    a_computed += 1
                else:
                    aids.add(s["aid"])
        rec["attack_ids"] = sorted(aids)
        rec["attack_sites"] = a_sites
        rec["attack_sites_computed"] = a_computed
        pairs[(m, sub)] = rec

    unattributed = collections.defaultdict(list)
    for s in sites:
        if s["eid"] is None or s["site"] in credited:
            continue
        unattributed[s["fn"]].append(
            dict(id=s["eid"], bone=s["bone"], frame=s["frame"],
                 site="0x%08X" % s["site"],
                 via="local" if s["local"] else VIA.get(s["via"], "?")))
    computed_sites = sum(1 for s in sites if s["eid"] is None)

    return dict(
        action_tick=None if entry is None else "0x%08X" % entry,
        main_states=mains,
        pairs=pairs,
        unattributed_effects=[
            dict(fn="0x%08X" % f, sites=sorted(v, key=lambda e: e["site"]))
            for f, v in sorted(unattributed.items())],
        effect_sites=len(sites),
        effect_sites_computed=computed_sites,
        attack_sites=len(atk_sites),
        attack_sites_uncredited=sum(1 for s in atk_sites
                                    if s["site"] not in atk_credited),
    )


# --------------------------------------------------------------------------- #
# the measured half
# --------------------------------------------------------------------------- #
def load_census(log: Path, since: int = 0) -> tuple[dict | None, str]:
    """`(census, reason)`. `census` is None whenever there is nothing to attach.

    The reason is written into the file verbatim, so a consumer that finds no
    measurements is told *why* rather than left to assume the run was clean.
    """
    if not log.exists():
        return None, "no log at %s" % log
    blob = log.read_bytes()[since:].decode("utf-8", "replace")
    lines = blob.splitlines()
    dwell, anims, moved = cs.census(lines)
    n_state = sum(1 for ln in lines if cs.RE_STATE.search(ln))
    if not dwell:
        return None, ("%s has %d [state] line(s) and no usable transitions — "
                      "the observe-only probe was not deployed for this run"
                      % (log, n_state))
    return dict(dwell=dwell, anims=anims, moved=moved,
                transitions=n_state, log=str(log), since=since,
                bytes=log.stat().st_size, mtime=int(log.stat().st_mtime)), ""


def measured_block(c: dict, key: tuple[int, int]) -> dict:
    """The measured half for one pair. `entered == 0` is a FINDING, not a gap."""
    d = c["dwell"].get(key, [])
    mv = c["moved"].get(key, [])
    out = dict(entered=len(d),
               dwell_ticks=round(sum(d) / len(d), 2) if d else 0.0,
               a1=sorted(c["anims"].get(key, ())),
               move_per_tick=round(sum(mv) / len(mv), 1) if mv else None,
               move_samples=len(mv))
    if not d:
        out["note"] = ("0 of %d observed transitions entered this pair — forced, "
                       "it bounces out in one tick" % c["transitions"])
    elif out["move_per_tick"] is None:
        # ⚠️ unmeasured movement is its own verdict and must not read as "still".
        out["note"] = ("never seen on two consecutive ticks with the monster "
                       "co-located, so movement is unmeasured, not zero")
    return out


# --------------------------------------------------------------------------- #
# the join
# --------------------------------------------------------------------------- #
def species_of(ov: Overlay) -> int:
    name = ov.name.split(".")[0]
    if not name.startswith("em") or not name[2:].isdigit():
        raise ValueError("%r is not an em<N>.ovl overlay" % ov.name)
    return int(name[2:])


# --------------------------------------------------------------------------- #
# the part system — the overlay's collision spheres, and the species damage grid
# --------------------------------------------------------------------------- #
GAME_TASK = "file_00070.bin"


def _sphere(s: hz.Sphere) -> dict:
    d = dict(bone=s.bone, shape=hz.CAPSULE if s.is_capsule else hz.SPHERE,
             hitzone_row=s.hitzone_row, part=s.part_index,
             radius=round(s.radius, 4), a=[round(v, 4) for v in s.a])
    if s.is_capsule:
        d["b"] = [round(v, 4) for v in s.b]
    if s.flags:
        d["flags"] = "0x%X" % s.flags
    return d


def parts_intel(ov: Overlay, species: int, game_task: Path | None) -> dict:
    """The two halves of "where can he be hit, and for how much".

    Both are STATIC — they are bytes in the ISO, like the moveset. Neither has ever
    been changed in a running game and verified (issue #19), so the block says that
    rather than implying the editor is authoring something proven to ship.
    """
    img = hz.Image.parse(Path(ov.path).read_bytes())
    sets = hz.find_sets(img)
    grid_img = (hz.Image.parse(game_task.read_bytes())
                if game_task and game_task.exists() else None)
    # Which set THIS species walks is not in the overlay at all: its game_task.ovl
    # row points at it (`+0x240`). The structural search finds most of them and
    # misses five (em01/15/40/58/82); the pointer walk is the authority, so a set
    # it names that `find_sets` did not is added rather than left out.
    own = hz.own_set(img, grid_img, species) if grid_img is not None else None
    owners = hz.species_sets(img, grid_img) if grid_img is not None else {}
    if own is not None and own.va not in {st.va for st in sets}:
        sets.append(own)
    out = {
        "present": True,
        "source": "tools/mhfu_model/hitzone.py",
        "note": "static: bytes in the ISO. NOT validated in game — no cold boot "
                "has ever changed either table and confirmed the effect (#19).",
        "sets": [
            {"va": "0x%08X" % st.va, "kind": st.kind, "count": len(st.spheres),
             "bones": st.bones, "parts": st.parts, "rows": st.rows,
             # the species id(s) whose row points at exactly this set. The overlay
             # serves several ids (em75: 75, 76, 81, 88 — one set each).
             "species": [sp for va, sp in sorted(owners.items()) if va == st.va],
             "spheres": [_sphere(s) for s in st.spheres]}
            for st in sorted(sets, key=lambda st: st.va)
            if st.kind != hz.KIND_UNKNOWN],
        "unclassified_runs": sum(1 for st in sets if st.kind == hz.KIND_UNKNOWN),
        # the set a weapon resolves against for THIS species, and the u32 that
        # says so — the in-place runtime seam and its capacity
        "active_set": None if own is None else "0x%08X" % own.va,
        "active_capacity": None if own is None else len(own.spheres),
        "sphere_table_field": ("0x%08X" % (hz.SPECIES_TABLE + hz.SPHERE_TABLE_FIELD
                                           + species * hz.SPECIES_STRIDE)),
        "grid": {"present": False, "reason": "%s not found beside the overlay"
                                             % GAME_TASK},
    }
    if grid_img is not None:
        g = hz.species_hitzones(grid_img, species)
        if g is None:
            out["grid"] = {"present": False,
                           "reason": "species %d has no hitzone state table at "
                                     "row+0x2FC" % species}
        else:
            out["grid"] = {
                "present": True,
                "file": GAME_TASK,
                "species_row": "0x%08X" % g.row_va,
                "state_table": "0x%08X" % g.state_table_va,
                "columns": list(hz.COLUMNS),
                "column_provenance": dict(hz.COLUMN_PROVENANCE),
                "element_bits": {k: "0x%X" % v for k, v in hz.ELEMENT_BITS.items()},
                "states": [{"va": "0x%08X" % b.va,
                            "rows": [list(r.values) for r in b.rows]}
                           for b in g.states],
                "note": "the grid is SHARED: it lives in species data, so a port "
                        "riding this host inherits it and editing it changes the "
                        "native monster too.",
            }
    return out


# --------------------------------------------------------------------------- #
# the attack system — where he hits YOU (issue #33)
# --------------------------------------------------------------------------- #
def _attack_sphere(s: hz.Sphere) -> dict:
    """Same shape as `_sphere` minus the two fields an attack volume does not use.
    The bone doubles as a coordinate space: 125/126/127 are the node's own."""
    d = dict(bone=s.bone, shape=hz.CAPSULE if s.is_capsule else hz.SPHERE,
             radius=round(s.radius, 4), a=[round(v, 4) for v in s.a])
    if s.is_capsule:
        d["b"] = [round(v, 4) for v in s.b]
    if s.flags:
        d["flags"] = "0x%X" % s.flags
    # kept so the round trip stays byte-identical even where the engine ignores them
    if s.hitzone_row or s.part:
        d["row_part"] = [s.hitzone_row, s.part]
    return d


def _attack_record(a: hb.Attack) -> dict:
    return dict(id=a.index, va="0x%08X" % a.va, power=a.power,
                element="0x%02X" % a.element, volume=a.volume, kind=a.kind,
                flags="0x%02X" % a.flags, angle=a.angle, tag="0x%02X" % a.tag,
                u16_0c=a.u16_0c, value_14=a.value_14, raw=a.raw.hex())


def attacks_intel(ov: Overlay, species: int, st: dict) -> dict:
    """The mirror image of `parts_intel`: the species' attack records and the
    volume sets they point at, plus WHICH game_task spawner its handlers call.

    STATIC — bytes in the ISO — but the join has two provenances and the block
    keeps them apart: the em75 spawner→table binding was walked live (#33); the
    other 16 are the id range fitting the biggest table (`em_attacks.consistency`).
    The runtime seam — each set overwritten IN PLACE through the overlay's own
    pointer table, sentinel-terminated — was proven by RAM poke on a native Tigrex
    (645 → 152 → 1381 units); the generated-Lua path is `mhfu_port.lua` P.hit().
    """
    img = hz.Image.parse(Path(ov.path).read_bytes())
    tables = hb.tables(img)
    prim = hb.primary_table(tables)
    sp = atk.spawner_of(ov)
    fit = atk.consistency(sp, prim)
    if not tables:
        return {"present": False,
                "reason": "this overlay never calls the table setter 0x%08X — no "
                          "attack table (em1/em33)" % hb.SETTER_VA,
                "spawner": None if sp is None else "0x%08X" % sp.fn}
    offsets = hb.ID_OFFSETS.get(species, {species: 0})

    def table_doc(t: hb.SpeciesTables) -> dict:
        return {
            "handle": "0x%08X" % t.handle_va,
            "records": "0x%08X" % t.records_va,
            "n_records": len(t.attacks),
            "volume_table": None if t.volume_table_va is None
            else "0x%08X" % t.volume_table_va,
            "n_sets": len(t.volumes),
            "primary": t is prim,
            # a table whose every set sits on 125/126/127 has no joint to draw on
            "rigged": any(hb.is_rigged(v.spheres) for v in t.volumes),
            "sets": [{"index": i, "va": "0x%08X" % v.va, "count": len(v.spheres),
                      "bones": v.bones, "rigged": hb.is_rigged(v.spheres),
                      "spheres": [_attack_sphere(s) for s in v.spheres]}
                     for i, v in enumerate(t.volumes)],
            "attacks": [_attack_record(a) for a in t.attacks],
        }

    return {
        "present": True,
        "source": "tools/mhfu_model/hitbox.py + tools/em_attacks.py",
        "note": "static: bytes in the ISO. The in-place set overwrite is proven by "
                "RAM poke on a native Tigrex (#33); the generated-Lua runtime path "
                "has not been cold-booted yet.",
        "setter": "0x%08X" % hb.SETTER_VA,
        "spawner": None if sp is None else "0x%08X" % sp.fn,
        "spawner_sites": 0 if sp is None else len(sp.sites),
        "spawner_literal_sites": 0 if sp is None else sp.n_literal,
        "join": fit,
        "join_provenance": ("measured — walked live from the handler to the HP "
                            "write, 2026-09-11" if fit == "measured" else
                            "inferred — the overlay's own game_task node "
                            "constructor, its literal ids against the biggest "
                            "table; see em_attacks.consistency"),
        "extra_spawners": [{"fn": "0x%08X" % ex.fn, "sites": len(ex.sites),
                            "ids": ex.literal_ids}
                           for ex in atk.extras(ov, sp)],
        "id_offsets": {str(k): v for k, v in sorted(offsets.items())},
        "field_provenance": dict(hb.FIELD_PROVENANCE),
        "attack_sites": st.get("attack_sites", 0),
        "attack_sites_uncredited": st.get("attack_sites_uncredited", 0),
        "tables": [table_doc(t) for t in tables],
    }


def build(path: Path, census: dict | None = None,
          census_reason: str = "", game_task: Path | None = None) -> dict:
    ov = Overlay.load_file(path)
    species = species_of(ov)
    st = static_intel(ov)
    have = census is not None

    chain = chn.chain(ov, species)
    prev: dict[tuple[int, int], list[list[int]]] = collections.defaultdict(list)
    for key, rec in chain["pairs"].items():
        for e in rec["next"]:
            for t in e["to"]:
                if list(key) not in prev[tuple(t)]:
                    prev[tuple(t)].append(list(key))

    pairs_out = []
    keys = set(st["pairs"])
    if have:
        keys |= set(census["dwell"])
    for key in sorted(keys):
        m, sub = key
        rec = dict(st["pairs"].get(key) or dict(main=m, sub=sub, handler=None))
        prov = {}
        if key in st["pairs"]:
            for f in ("handler", "a1", "ends_on", "event_frames", "window_frames",
                      "effects",
                      "budget"):
                if f in rec:
                    prov[f] = STATIC
            prov.setdefault("handler", STATIC)
        if key in chain["pairs"]:
            # the hand-off: every enter-action site the handler can reach, with the
            # guards on the path. An EMPTY list on a handled pair is a finding too —
            # the handler never ends the action itself (the brain has to).
            rec["next"] = chain["pairs"][key]["next"]
            rec["prev"] = sorted(prev.get(key, []))
            prov["next"] = STATIC
            prov["prev"] = STATIC
        else:
            # the census saw a pair the dispatcher walk did not enumerate
            rec["note"] = ("not in the overlay's (main,sub) jump tables — the "
                           "engine reached it by a path this extractor cannot see")
        if have:
            rec["measured"] = measured_block(census, key)
            for f in ("entered", "dwell_ticks", "move_per_tick"):
                prov[f] = MEASURED
            if rec["measured"]["a1"]:
                prov["measured_a1"] = MEASURED
        else:
            rec["measured"] = None
            for f in ("entered", "dwell_ticks", "move_per_tick"):
                prov[f] = ABSENT
        rec["provenance"] = prov
        pairs_out.append(rec)

    blob = Path(ov.path).read_bytes()
    doc = {
        "schema": SCHEMA,
        "host_species": species,
        "generated_by": "tools/em_intel.py",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "overlay": {
            "name": ov.name,
            "file": Path(ov.path).name,
            "sha1": hashlib.sha1(blob).hexdigest(),
            "load": "0x%08X" % ov.load,
            "text": ["0x%08X" % ov.text_va, "0x%08X" % ov.text_end],
            "region": "MHFU EU (ULES01213)",
            "effect_id_bias": fx.SPECIES_BIAS.get(species, 0),
        },
        "action_tick": st["action_tick"],
        "main_states": st["main_states"],
        "static": {
            "present": True,
            "source": "overlay disassembly",
            "tools": ["em_moveset.py", "em_phase_map.py", "em_effects.py"],
            "note": "a property of the ISO: the same for every run and every "
                    "player. Never a measurement.",
            "effect_call_depth": EFFECT_CALL_DEPTH,
            "effect_sites": st["effect_sites"],
            "effect_sites_computed": st["effect_sites_computed"],
        },
        "census": {
            "present": have,
            "source": "tools/em_state_census.py",
            "note": "the ONLY source that knows whether a pair is usable: 411 of "
                    "411 forced moves into never-entered pairs survived exactly "
                    "one tick.",
        },
        "pairs": pairs_out,
        "chain": {
            "source": "tools/em_chain.py",
            "enter_action": chain["enter_action"],
            "species_byte": chain["species"],
            "note": "a handler ends an action by calling enter-action (vt+0x88) "
                    "with a literal (main, id); the per-main translator turns the "
                    "id into the pair AND provisions the handler (the charge's run "
                    "budget +0x76C is set there, not by act_set). `next` is that "
                    "call, read statically, with the guards on the path; `prev` is "
                    "its inverse. Pairs with no `next` never end themselves.",
            "hubs": chain_hubs(chain),
            "brain": chain["brain"],
            "translators": chain["translators"],
        },
        "unattributed_effects": st["unattributed_effects"],
        "parts": parts_intel(ov, species, game_task),
        "attacks": attacks_intel(ov, species, st),
    }
    if have:
        doc["census"].update(log=census["log"], since=census["since"],
                             log_bytes=census["bytes"],
                             log_mtime=census["mtime"],
                             transitions=census["transitions"],
                             observed_pairs=len(census["dwell"]),
                             attributed_by="--census-species (the log does not "
                                           "record which species it watched)")
    else:
        doc["census"].update(
            reason=census_reason or "no census supplied",
            how="deploy the observe-only probe, cold-boot into a quest with this "
                "species, then re-run with --log <framework.log>",
            consequence="every pair's `measured` block is null and `entered` is "
                        "UNKNOWN — not zero. A consumer must not treat an absent "
                        "measurement as a never-entered pair.")
    return doc


def chain_hubs(chain: dict, min_in: int = 8) -> list[list[int]]:
    """The pairs most hand-offs land in — where the brain thinks again. em75: (0,1)
    and (0,2) (alert/idle), (2,2) (the +0x280 reaction) and (0,3) (the run's
    stop). A graph view collapses these into terminals, or every chain is one
    arrow into the same three boxes.

    ⚠️ Counted per HANDLER, not per pair: em75's 48 main-3 subs share one handler
    whose union of hand-offs would otherwise vote 48 times for its own siblings."""
    cnt = collections.Counter()
    by_handler: dict[str, set] = collections.defaultdict(set)
    for rec in chain["pairs"].values():
        for e in rec["next"]:
            for t in e["to"]:
                by_handler[rec["handler"]].add(tuple(t))
    for targets in by_handler.values():
        for t in targets:
            cnt[t] += 1
    return [list(k) for k, n in cnt.most_common() if n >= min_in]


def summarise(doc: dict) -> str:
    pairs = doc["pairs"]
    handled = [p for p in pairs if p.get("handler")]
    eff = [p for p in handled if p.get("effects")]
    budget = [p for p in handled if (p.get("budget") or {}).get("gated")]
    owned = [p for p in budget if p["budget"]["post_hook_owns"]]
    ends = collections.Counter(p.get("ends_on", "-") for p in handled)
    n_un = sum(len(u["sites"]) for u in doc["unattributed_effects"])
    chained = [p for p in handled if p.get("next")]
    resolved = [p for p in chained if any(e["to"] for e in p["next"])]
    out = [
        "%s  species %d  %d pair(s), %d with a handler"
        % (doc["overlay"]["name"], doc["host_species"], len(pairs), len(handled)),
        "  ends on: " + ", ".join("%s=%d" % kv for kv in sorted(ends.items())),
        "  hands off: %d pair(s), %d to a resolved pair; hubs %s  (STATIC)"
        % (len(chained), len(resolved),
           " ".join("(%d,%d)" % tuple(h) for h in doc["chain"]["hubs"]) or "-"),
        "  %d pair(s) carry effects (%d site(s) unattributed in %d function(s))"
        % (len(eff), n_un, len(doc["unattributed_effects"])),
        "  %d budget-gated, %d of them ownable by a slot-32 post-hook"
        % (len(budget), len(owned)),
    ]
    pt = doc.get("parts") or {}
    hb = [s for s in pt.get("sets", []) if s["kind"] == hz.KIND_HURTBOX]
    g = pt.get("grid") or {}
    out.append("  parts: %d hurtbox set(s), %d sphere(s); grid %s"
               % (len(hb), sum(s["count"] for s in hb),
                  "%d state(s)" % len(g["states"]) if g.get("present")
                  else "ABSENT (%s)" % g.get("reason", "?")))
    at = doc.get("attacks") or {}
    if at.get("present"):
        prim = next((t for t in at["tables"] if t["primary"]), None)
        out.append("  attacks: %d table(s); moveset %s records / %s set(s); spawner "
                   "%s (%s), %d pair(s) name an attack"
                   % (len(at["tables"]),
                      "-" if prim is None else prim["n_records"],
                      "-" if prim is None else prim["n_sets"],
                      at.get("spawner") or "-", at.get("join"),
                      sum(1 for p in pairs if p.get("attack_ids"))))
    else:
        out.append("  attacks: ABSENT (%s)" % at.get("reason", "?"))
    c = doc["census"]
    if c["present"]:
        out.append("  census: %d transitions, %d pair(s) observed  (MEASURED)"
                   % (c["transitions"], c["observed_pairs"]))
    else:
        out.append("  census: ABSENT — %s" % c["reason"])
    return "\n".join(out)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("overlay", nargs="*",
                    help="em*.ovl file(s), by path or by name")
    ap.add_argument("--all", action="store_true", help="every em*.ovl (17)")
    ap.add_argument("--out", default=str(OUT_ROOT),
                    help="output directory (default: species/)")
    ap.add_argument("--stdout", action="store_true",
                    help="print the JSON instead of writing it")
    ap.add_argument("--log", default=str(cs.DEFAULT_LOG),
                    help="framework.log to take the census from")
    ap.add_argument("--since", type=int, default=0,
                    help="byte offset — framework.log spans many boots")
    ap.add_argument("--census-species", type=int,
                    help="which species the log's [state] lines came from. "
                         "REQUIRED to attach a census to more than one overlay: "
                         "the log does not record it, and attaching a Tigrex "
                         "census to another species would be a measured lie.")
    ap.add_argument("--no-census", action="store_true",
                    help="never attach measurements, even if a log has them")
    ap.add_argument("--game-task", default="",
                    help="game_task.ovl (file_00070.bin) — the species damage "
                         "grid. Defaults to the overlay's own directory.")
    ap.add_argument("--quiet", "-q", action="store_true",
                    help="write the files without the per-species summary")
    a = ap.parse_args(argv)

    if a.all or not a.overlay:
        paths = fx.em_overlays()
    else:
        paths = [Path(x) if Path(x).exists() else fx.DATA_DIR / x
                 for x in a.overlay]
    if not paths:
        print("no em*.ovl overlays found under %s — extract your own ISO first "
              "(docs/ASSETS.md)" % fx.DATA_DIR, file=sys.stderr)
        return 1

    if a.no_census:
        census, reason = None, "--no-census"
    else:
        census, reason = load_census(Path(a.log), a.since)
    target = a.census_species
    if census is not None and target is None:
        if len(paths) == 1:
            target = species_of(Overlay.load_file(paths[0]))
        else:
            if not a.quiet:
                print("refusing to attach the census in %s to %d overlays: it does "
                      "not record which species it watched. Pass --census-species."
                      % (a.log, len(paths)), file=sys.stderr)
            census, reason = None, ("a census was found in %s but not attached: "
                                    "--census-species was not given" % a.log)

    outdir = Path(a.out)
    rc = 0
    for p in paths:
        try:
            ov = Overlay.load_file(p)
            sp = species_of(ov)
        except Exception as e:
            print("%s: not an em overlay (%s)" % (p, e), file=sys.stderr)
            rc = 1
            continue
        mine = census if (census is not None and sp == target) else None
        why = reason
        if census is not None and mine is None:
            why = ("the census in %s was taken from species %s, not %d"
                   % (a.log, target, sp))
        try:
            gt = Path(a.game_task) if a.game_task else Path(p).parent / GAME_TASK
            doc = build(p, mine, why, gt)
        except Exception as e:                                # pragma: no cover
            print("%s: %s" % (p, e), file=sys.stderr)
            rc = 1
            continue
        text = json.dumps(doc, indent=1, sort_keys=False)
        if a.stdout:
            print(text)
        else:
            outdir.mkdir(parents=True, exist_ok=True)
            dest = outdir / ("em%02d.json" % doc["host_species"])
            dest.write_text(text + "\n", encoding="utf-8")
            if not a.quiet:
                print(summarise(doc))
                print("  -> %s (%.1f KB)" % (dest, len(text) / 1024.0))
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
