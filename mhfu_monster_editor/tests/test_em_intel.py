"""`tools/em_intel.py` — the join, and the promise that it invents nothing.

The generator's value rests on one thing: a number in `species/emNN.json` means what
the analyser it came from meant. So the tests that matter are equalities against those
analysers, run on the real overlay, not assertions about the shape of the output.

Two halves:

* the census plumbing is pure and runs anywhere — a synthetic `framework.log` is written
  to a temp file and parsed, so the "measured" path is exercised on every machine even
  though no measured data exists in the repo;
* the join itself needs `workspace/extracted/data_files` and RETURNS EARLY without it,
  like `test_ports_build.py`. No game data lives in this repository.

🔴 The anti-fabrication test is `test_no_census_leaves_every_measured_block_null`. With
no log, every `measured` must be `null` and every `entered` provenance `absent`. A
generator that helpfully defaulted `entered` to 0 would turn "we did not look" into
"the engine never goes there", which is the one finding that rejects a bind.
"""
import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

from mhfu_monster_editor import intel as I

import em_intel as G
import em_effects as fx
import em_moveset as mvs
import em_phase_map as pm
from ovl_explore import Overlay

DATA = os.path.join(_ROOT, "workspace", "extracted", "data_files")
EM75 = os.path.join(DATA, "file_06108.bin")


def _have_data():
    return os.path.exists(EM75)


# --------------------------------------------------------------------------- #
# the census plumbing — no game data needed
# --------------------------------------------------------------------------- #
SYNTHETIC = []
_t = 0
for _pair_, _dwell, _a1 in [((2, 8), 24, 15), ((0, 7), 40, 3), ((3, 6), 1, 9),
                            ((2, 8), 22, 15), ((0, 7), 36, 3), ((2, 8), 25, 14)]:
    SYNTHETIC.append("[state] main=%d sub=%d (a1=%d) t=%d"
                     % (_pair_[0], _pair_[1], _a1, _t))
    for _k in range(_dwell):
        SYNTHETIC.append("[brute] t=%d sec=3/3 SAME out=1 in=1 x=0 d=%d"
                         % (_t + _k, 500 + 40 * _k))
    _t += _dwell
SYNTHETIC.append("[state] main=2 sub=8 (a1=15) t=%d" % _t)


def _log(lines):
    fh = tempfile.NamedTemporaryFile("w", suffix=".log", delete=False)
    fh.write("\n".join(lines) + "\n")
    fh.close()
    return fh.name


def test_a_missing_log_is_not_an_error_it_is_a_stated_reason():
    c, why = G.load_census(G.Path("/nonexistent/framework.log"))
    assert c is None and "no log at" in why


def test_a_log_with_no_state_lines_says_the_probe_was_not_deployed():
    """This is the repo's actual condition: framework.log exists and has none."""
    p = _log(["[boot] framework up", "[lua] tick 1"])
    try:
        c, why = G.load_census(G.Path(p))
        assert c is None
        assert "[state]" in why and "probe" in why
    finally:
        os.unlink(p)


def test_a_real_log_produces_dwell_that_matches_em_state_census_itself():
    p = _log(SYNTHETIC)
    try:
        c, why = G.load_census(G.Path(p))
        assert c is not None and why == ""
        dwell, anims, moved = __import__("em_state_census").census(SYNTHETIC)
        assert c["dwell"] == dwell and c["anims"] == anims
        assert c["transitions"] == 7
        b = G.measured_block(c, (2, 8))
        assert b["entered"] == 3 and abs(b["dwell_ticks"] - 23.67) < 0.01
        assert b["a1"] == [14, 15]
    finally:
        os.unlink(p)


def test_a_pair_the_census_never_saw_is_recorded_as_a_measured_zero_with_the_sample():
    """`entered: 0` is a claim, so the file has to say how big the sample was."""
    p = _log(SYNTHETIC)
    try:
        c, _ = G.load_census(G.Path(p))
        b = G.measured_block(c, (4, 15))
        assert b["entered"] == 0 and b["dwell_ticks"] == 0.0
        assert "0 of 7 observed transitions" in b["note"]
    finally:
        os.unlink(p)


def test_unmeasured_movement_is_null_and_says_so_rather_than_reading_as_zero():
    """⚠️ `move_per_tick = 0` would mean STATIONARY. `None` means nobody looked —
    the distinction that made (0,7), the longest-dwelling state in the table, look
    like a poor candidate on no evidence at all."""
    p = _log(["[state] main=1 sub=1 (a1=2) t=0", "[state] main=1 sub=2 (a1=3) t=4",
              "[state] main=1 sub=1 (a1=2) t=9"])
    try:
        c, _ = G.load_census(G.Path(p))
        b = G.measured_block(c, (1, 1))
        assert b["entered"] == 1 and b["move_per_tick"] is None
        assert "unmeasured, not zero" in b["note"]
    finally:
        os.unlink(p)


def test_the_ends_on_classification_is_em_phase_maps_own_precedence():
    """clip-done wins when present: the handler cannot advance past that phase until
    the clip stops playing, whatever else it also tests."""
    def g(clip=0, timer=0, frames=(), windows=()):
        return dict(clip_done=clip, timer=timer, frames=list(frames),
                    windows=list(windows), phases=set())
    assert G.ends_on(g(clip=1)) == "clip"
    assert G.ends_on(g(clip=1, frames=[60.0])) == "clip+cursor"
    assert G.ends_on(g(clip=2, timer=1, frames=[60.0])) == "clip+cursor"
    assert G.ends_on(g(timer=1)) == "budget"
    assert G.ends_on(g(frames=[60.0])) == "cursor"
    assert G.ends_on(g(windows=[1])) == "cursor"
    assert G.ends_on(g()) == "unknown"


# --------------------------------------------------------------------------- #
# the join, against the real overlay
# --------------------------------------------------------------------------- #
def test_em75_joins_every_dispatcher_case_and_names_the_ones_with_no_handler():
    if not _have_data():
        return
    doc = G.build(G.Path(EM75), None, "unit test")
    assert doc["schema"] == I.SCHEMA and doc["host_species"] == 75
    assert doc["action_tick"] == "0x09D36A08"
    assert len(doc["main_states"]) == 8
    handled = [p for p in doc["pairs"] if p["handler"]]
    assert len(doc["pairs"]) == 242 and len(handled) == 231
    # mains 5/6/7 dispatch some way the extractor does not handle. Saying "not
    # enumerated" is a different claim from "no actions", and the file must make it.
    unenum = [m["main"] for m in doc["main_states"] if not m["enumerated"]]
    assert unenum == [5, 6, 7]
    for m in doc["main_states"]:
        if not m["enumerated"]:
            assert m["sub_states"] is None and "not enumerable" in m["note"]


def test_no_census_leaves_every_measured_block_null():
    """🔴 The anti-fabrication test. Nothing may be defaulted, estimated or filled in."""
    if not _have_data():
        return
    doc = G.build(G.Path(EM75), None, "unit test")
    assert doc["census"]["present"] is False
    assert doc["census"]["reason"] == "unit test"
    assert "not treat an absent measurement" in doc["census"]["consequence"]
    assert all(p["measured"] is None for p in doc["pairs"])
    assert all(p["provenance"]["entered"] == G.ABSENT for p in doc["pairs"])
    assert not any("entered" in p for p in doc["pairs"]), "no top-level entered key"
    si = I.SpeciesIntel.from_dict(doc)
    assert si.has_census is False and si.has_static is True
    assert all(p.entered is None and not p.never_entered for p in si)


def test_every_static_field_is_the_analyser_it_came_from():
    if not _have_data():
        return
    ov = Overlay.load_file(EM75)
    doc = G.build(G.Path(EM75), None, "")
    si = I.SpeciesIntel.from_dict(doc)
    hs = pm.handlers(ov)
    assert len(hs) == 231
    # the generator walks the dispatcher itself (it needs the per-main bias and case
    # count that `handlers()` throws away), so pin the two walks against each other —
    # a divergence here is how the join would silently point at the wrong function.
    mine = {(p["main"], p["sub"]): int(p["handler"], 0)
            for p in doc["pairs"] if p["handler"]}
    assert mine == hs, "the case walk diverged from em_phase_map.handlers"
    checked = 0
    for (m, s), fn in sorted(hs.items())[::17]:            # a spread, not all 231
        p = si.pair(m, s)
        assert p is not None and p.handler == fn, (m, s)
        gates = pm.analyse(ov, fn)
        assert p.ends_on == G.ends_on(gates)
        assert p.windows == len(gates["windows"])
        a1s, computed = mvs.exec_consts(ov, fn)
        assert p.a1_static == sorted(a1s) and p.a1_computed == computed
        if p.budget.gated:
            assert p.budget.phase0_seeds == pm.budget_owner(ov, fn)
            assert p.budget.post_hook_owns == (not p.budget.phase0_seeds)
        checked += 1
    assert checked >= 10


def test_the_budget_verdict_matches_em_phase_map_budget_owner_over_all_27():
    """(2,24) and (2,9) are owned live, (2,16) is not — EM_OVERLAY_ABI §13."""
    if not _have_data():
        return
    si = I.SpeciesIntel.from_dict(G.build(G.Path(EM75), None, ""))
    gated = si.budget_gated()
    assert len(gated) == 27
    assert sum(1 for p in gated if p.budget.post_hook_owns) == 15
    assert si.pair(2, 24).budget.post_hook_owns is True
    assert si.pair(2, 9).budget.post_hook_owns is True
    assert si.pair(2, 16).budget.post_hook_owns is False
    assert si.pair(2, 16).budget.phase0_seeds == [150]


def test_every_literal_effect_site_is_either_credited_to_a_pair_or_listed_apart():
    """em75's richest effect routines hang off a species-byte switch no pair handler
    calls. Spreading them across pairs would be a guess; dropping them would lose the
    species' whole vocabulary. They go in `unattributed_effects`, counted."""
    if not _have_data():
        return
    ov = Overlay.load_file(EM75)
    literal = [s for s in fx.spawns(ov) if s["eid"] is not None]
    assert len(literal) == 52
    doc = G.build(G.Path(EM75), None, "")
    credited = {e["site"] for p in doc["pairs"] for e in p.get("effects", [])}
    apart = {e["site"] for u in doc["unattributed_effects"] for e in u["sites"]}
    assert credited & apart == set(), "a site cannot be both"
    assert len(credited | apart) == len(literal), "a literal site went missing"
    assert len(credited) == 11 and len(apart) == 41
    # 11 sites, 14 attributions: several (main,sub) share one handler, so the same
    # spawn site legitimately belongs to more than one pair.
    assert sum(len(p.get("effects", [])) for p in doc["pairs"]) == 14
    assert doc["static"]["effect_sites_computed"] == 3


def test_the_written_file_round_trips_through_the_editors_reader():
    if not _have_data():
        return
    with tempfile.TemporaryDirectory() as td:
        rc = G.main(["file_06108.bin", "--out", td, "--no-census", "-q"])
        assert rc == 0
        si = I.find_intel(75, root=td)
        assert si is not None and len(si) == 242
        p = si.pair(0, 19)
        assert p.a1 == [97] and p.ends_on == "budget"
        assert [str(e) for e in p.effects] == ["79@b37", "85@b37", "87@b37", "87@b37"]
        assert si.bindable(0, 19).unverified is True
        assert I.find_intel(17, root=td) is None


def test_a_census_only_attaches_to_the_species_it_was_taken_from():
    """⚠️ `em_state_census` does not record which species produced its [state] lines.
    Attaching a Tigrex log to all 17 species would write measured lies into 16 files."""
    if not _have_data():
        return
    p = _log(SYNTHETIC)
    try:
        with tempfile.TemporaryDirectory() as td:
            rc = G.main(["file_06108.bin", "file_06099.bin", "--log", p,
                         "--census-species", "75", "--out", td, "-q"])
            assert rc == 0
            got = I.find_intel(75, root=td)
            other = I.find_intel(17, root=td)
            assert got.has_census is True and other.has_census is False
            assert "species 75, not 17" in other.census_reason
            assert got.pair(2, 8).entered == 3
            assert got.pair(4, 15).entered == 0        # measured zero, a finding
            assert not got.bindable(4, 15).ok
            assert got.bindable(4, 15, override=True).ok
    finally:
        os.unlink(p)


def test_more_than_one_overlay_with_a_log_and_no_species_refuses_to_guess():
    if not _have_data():
        return
    p = _log(SYNTHETIC)
    try:
        with tempfile.TemporaryDirectory() as td:
            assert G.main(["file_06108.bin", "file_06099.bin", "--log", p,
                           "--out", td, "-q"]) == 0
            for sp in (75, 17):
                si = I.find_intel(sp, root=td)
                assert si.has_census is False
                assert "--census-species" in si.census_reason
    finally:
        os.unlink(p)


def test_all_seventeen_overlays_build():
    if not _have_data():
        return
    paths = fx.em_overlays()
    assert len(paths) == 17, [p.name for p in paths]
    with tempfile.TemporaryDirectory() as td:
        assert G.main(["--all", "--out", td, "--no-census", "-q"]) == 0
        written = sorted(os.listdir(td))
        assert len(written) == 17, written
        for name in written:
            doc = json.loads(open(os.path.join(td, name)).read())
            si = I.SpeciesIntel.from_dict(doc)
            assert doc["overlay"]["name"] == "em%02d.ovl" % si.host_species
            assert si.host_species == int(name[2:-5])
            assert len(si) > 0 and si.has_static
            assert doc["action_tick"], name
            assert doc["overlay"]["region"] == "MHFU EU (ULES01213)"



def test_em75_attack_sites_are_credited_or_counted_apart_and_the_charge_resolves():
    """Same discipline as the effects: every literal attack-spawn site is either
    reached from a pair handler or counted as uncredited; none is invented. And the
    one pair walked live, (1,4), names attack 6 — set 2, the set the RAM poke moved."""
    if not _have_data():
        return
    doc = G.build(G.Path(EM75), None, "test", game_task=None)
    at = doc["attacks"]
    assert at["present"] and at["spawner"] == "0x09B661E8" and at["join"] == "measured"
    credited = sum(p.get("attack_sites", 0) for p in doc["pairs"])
    assert at["attack_sites"] == 84, at["attack_sites"]
    assert credited > 0 and at["attack_sites_uncredited"] < 84
    p14 = next(p for p in doc["pairs"] if (p["main"], p["sub"]) == (1, 4))
    assert p14["attack_ids"] == [6, 31], p14["attack_ids"]
    prim = next(t for t in at["tables"] if t["primary"])
    assert prim["n_records"] == 107 and prim["n_sets"] == 56
    assert prim["attacks"][6]["power"] == 64 and prim["attacks"][6]["volume"] == 2
    assert prim["sets"][2]["count"] == 10
    assert at["id_offsets"] == {"75": 0, "76": 33, "88": 70}
    # the reader agrees with the writer
    si = I.SpeciesIntel.from_dict(doc)
    assert si.attacks.sets_for(si.pair(1, 4).attack_ids) == [2]


def test_every_overlay_states_its_attack_join_provenance():
    """em1 and em33 have no table and say so; the rest name a spawner and whether
    the id range fits the table read — a stated inconsistency, never a silent one."""
    if not _have_data():
        return
    paths = fx.em_overlays()
    seen = {}
    for p in paths:
        doc = G.build(p, None, "test", game_task=None)
        at = doc["attacks"]
        seen[doc["host_species"]] = at.get("join") if at["present"] else "absent"
    assert seen[1] == "absent" and seen[33] == "absent", seen
    assert seen[75] == "measured"
    for sp in (7, 14, 17, 21, 40, 58, 83):
        assert seen[sp] == "consistent", (sp, seen[sp])
    for sp in (2, 15, 20, 55, 59, 82):
        assert seen[sp] == "ids_exceed_table", (sp, seen[sp])

if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not _have_data():
        print("  workspace/ extracts absent — the overlay join tests are skipped "
              "(docs/ASSETS.md)")
    bad = 0
    for f in fns:
        try:
            f()
            print("  PASS  %s" % f.__name__)
        except Exception as e:
            bad += 1
            print("  FAIL  %s: %s" % (f.__name__, e))
    print("\n%d tests, %d failed" % (len(fns), bad))
    sys.exit(1 if bad else 0)
