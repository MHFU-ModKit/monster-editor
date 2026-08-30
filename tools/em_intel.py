#!/usr/bin/env python3
"""Join every offline analyser for one species into `species/emNN.json`.

Four tools already answer "what does the host action expect?", each in its own
format and none of them joined:

    em_moveset.py <ovl> --states   every reachable (main,sub) and its a1 clip ids
    em_phase_map.py <ovl>          what ENDS the action: clip-done / cursor
                                   frames / the +0x414 budget (and who owns it)
    em_effects.py <ovl>            per-handler effect recipes (id @ bone @ frame)
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


def build(path: Path, census: dict | None = None,
          census_reason: str = "") -> dict:
    ov = Overlay.load_file(path)
    species = species_of(ov)
    st = static_intel(ov)
    have = census is not None

    pairs_out = []
    keys = set(st["pairs"])
    if have:
        keys |= set(census["dwell"])
    for key in sorted(keys):
        m, sub = key
        rec = dict(st["pairs"].get(key) or dict(main=m, sub=sub, handler=None))
        prov = {}
        if key in st["pairs"]:
            for f in ("handler", "a1", "ends_on", "event_frames", "effects",
                      "budget"):
                if f in rec:
                    prov[f] = STATIC
            prov.setdefault("handler", STATIC)
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
        "unattributed_effects": st["unattributed_effects"],
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


def summarise(doc: dict) -> str:
    pairs = doc["pairs"]
    handled = [p for p in pairs if p.get("handler")]
    eff = [p for p in handled if p.get("effects")]
    budget = [p for p in handled if (p.get("budget") or {}).get("gated")]
    owned = [p for p in budget if p["budget"]["post_hook_owns"]]
    ends = collections.Counter(p.get("ends_on", "-") for p in handled)
    n_un = sum(len(u["sites"]) for u in doc["unattributed_effects"])
    out = [
        "%s  species %d  %d pair(s), %d with a handler"
        % (doc["overlay"]["name"], doc["host_species"], len(pairs), len(handled)),
        "  ends on: " + ", ".join("%s=%d" % kv for kv in sorted(ends.items())),
        "  %d pair(s) carry effects (%d site(s) unattributed in %d function(s))"
        % (len(eff), n_un, len(doc["unattributed_effects"])),
        "  %d budget-gated, %d of them ownable by a slot-32 post-hook"
        % (len(budget), len(owned)),
    ]
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
            doc = build(p, mine, why)
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
