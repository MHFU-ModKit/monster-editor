"""The action inspector's join: the host's frames against the port's clip.

Every test here runs on synthetic intel and a synthetic manifest — no game data, no GL,
no display. That is the point: the alignment IS the feature, and the arithmetic that
decides "your impact is 19 frames early" should be checkable on any machine. Two tests
at the end use the real `species/em75.json` when it has been generated, and return
early when it has not.

    venv/bin/python mhfu_monster_editor/tests/test_align.py
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mhfu_monster_editor import align as A
from mhfu_monster_editor import manifest as MF
from mhfu_monster_editor.intel import SpeciesIntel

BASE = """
[port]
name = "t"
host_species = 75
pac = "t.bin"

[source]
model = 5339

[clips.charge]
slot = 61
frames = 382
loop = false
impact_frame = 41

[moves.charge]
main = 2
sub = 8
clip = "charge"
"""


def _m(extra: str = "") -> MF.PortManifest:
    return MF.loads(BASE + extra)


def _intel(**pair) -> SpeciesIntel:
    """One-pair intel in the joined shape `tools/em_intel.py` writes."""
    rec = {"main": 2, "sub": 8, "handler": "0x09D34390", "a1": [15],
           "ends_on": "clip+cursor", "event_frames": [], "window_frames": [],
           "windows": 0, "budget": {"gated": False}, "effects": [],
           "measured": None}
    rec.update(pair)
    return SpeciesIntel.from_dict({"host_species": 75, "pairs": [rec],
                                   "static": {"present": True},
                                   "census": {"present": bool(rec["measured"])}})


RIG = A.PortRig(n_bones=47, driven=set(range(44)), vertices={33: 120, 45: 6})


# --------------------------------------------------------------------------- #
# the headline — the sentence issue #9 exists for
# --------------------------------------------------------------------------- #
def test_the_headline_puts_the_handlers_frame_next_to_yours():
    """🔴 The Done-when, verbatim: "this handler tests frame 60; your clip's impact is
    at frame 41"."""
    al = A.align(_m(), "charge", _intel(event_frames=[60.0]))
    assert "tests frame 60" in al.headline, al.headline
    assert "impact is at frame 41" in al.headline, al.headline
    assert "19.0 frame(s) early (the nearest test is 60)" in al.headline, al.headline
    assert any(f.code == "IMPACT_OFF_GATE" for f in al.warnings), al.findings
    print("headline          %s" % al.headline)


def test_an_impact_on_the_gate_is_reported_as_aligned():
    al = A.align(_m(), "charge", _intel(event_frames=[40.0, 90.0]))
    assert "on the 40 test (+1.0)" in al.headline, al.headline
    assert not [f for f in al.findings if f.code == "IMPACT_OFF_GATE"]
    print("headline          within %g frames counts as aligned"
          % A.IMPACT_TOLERANCE)


def test_no_impact_frame_asks_for_one_instead_of_guessing():
    m = _m()
    m.clips["charge"].impact_frame = None
    al = A.align(m, "charge", _intel(event_frames=[60.0]))
    assert "records no impact frame" in al.headline, al.headline
    assert al.impact is None
    print("headline          no impact frame -> asks for one")


def test_a_pair_with_no_fixed_frames_says_only_length_matters():
    al = A.align(_m(), "charge", _intel())
    assert "no fixed frames" in al.headline, al.headline
    assert any(f.code == "NO_FIXED_FRAMES" for f in al.findings)
    print("headline          a handler that tests nothing says so")


# --------------------------------------------------------------------------- #
# gates
# --------------------------------------------------------------------------- #
def test_a_gate_past_the_clips_last_frame_is_an_error_not_a_note():
    """🔴 The cursor never reaches it, so that branch of the handler never runs — and
    a clip too short for its action is silent on screen."""
    al = A.align(_m(), "charge", _intel(event_frames=[60.0, 500.0]),
                 clip_frames=382)
    e = [f for f in al.errors if f.code == "GATE_BEYOND_CLIP"]
    assert e, al.findings
    assert "500" in e[0].message and "382" in e[0].message, e[0].message
    assert [mk.frame for mk in al.markers if mk.unreachable] == [500.0]
    print("gates             a gate past the clip end -> ERROR, and drawn unreachable")


def test_the_windowed_form_lands_on_the_timeline_too():
    """`0x08864348` is the hitbox-active shape — 280 sites in em75, and useless as a
    bare count. The literals are markers of their own kind."""
    al = A.align(_m(), "charge", _intel(event_frames=[40.0],
                                        window_frames=[-5.0, 58.0, 58.0], windows=3))
    kinds = {mk.kind for mk in al.markers}
    assert A.WINDOW in kinds and A.GATE in kinds, kinds
    assert al.gates == [-5.0, 40.0, 58.0], al.gates
    # deduplicated: 58 appears three times in the overlay and once on the timeline
    assert sum(1 for mk in al.markers if mk.frame == 58.0) == 1
    print("gates             window literals become their own markers, deduplicated")


def test_the_clip_frames_argument_beats_the_manifests_own_number():
    """A rebuild moves clips; "the handler tests 60" against a stale length is the
    wrong answer, so the BUILD's length wins when it is supplied."""
    al = A.align(_m(), "charge", _intel(event_frames=[300.0]), clip_frames=120)
    assert al.frames == 120, al.frames
    assert any(f.code == "GATE_BEYOND_CLIP" for f in al.errors), al.findings
    print("gates             the build's clip length wins over the manifest's")


# --------------------------------------------------------------------------- #
# what ends the action
# --------------------------------------------------------------------------- #
def test_an_action_that_ends_with_the_clip_says_the_length_is_free():
    al = A.align(_m(), "charge", _intel(ends_on="clip"))
    assert any(f.code == "ENDS_ON_CLIP" and f.level == A.INFO for f in al.findings)
    print("ends on           clip-done -> the length is yours (182 of 231 are)")


def test_a_budget_gated_action_warns_and_names_who_can_set_the_budget():
    al = A.align(_m(), "charge",
                 _intel(ends_on="budget",
                        budget={"gated": True, "post_hook_owns": True}))
    f = [x for x in al.warnings if x.code == "ENDS_ON_BUDGET"]
    assert f and "slot-32" in f[0].message, al.findings
    reseeded = A.align(_m(), "charge",
                       _intel(ends_on="budget",
                              budget={"gated": True, "post_hook_owns": False,
                                      "phase0_seeds": [150]}))
    g = [x for x in reseeded.warnings if x.code == "ENDS_ON_BUDGET"]
    assert "RE-SEEDS" in g[0].message and "150" in g[0].message, g[0].message
    print("ends on           +0x414 budget -> warns, and says who can own it")


# --------------------------------------------------------------------------- #
# the census — the one loud finding
# --------------------------------------------------------------------------- #
def test_a_pair_the_census_measured_as_never_entered_is_an_error():
    al = A.align(_m(), "charge",
                 _intel(measured={"entered": 0, "dwell_ticks": 0.0}))
    e = [f for f in al.errors if f.code == "NEVER_ENTERED"]
    assert e, al.findings
    assert "411 of 411" in e[0].message
    print("census            measured zero entries -> ERROR")


def test_allow_unentered_downgrades_it_to_the_authors_call():
    al = A.align(_m('\n[moves.risky]\nmain = 4\nsub = 15\nclip = "charge"\n'
                    'allow_unentered = true\n'), "risky",
                 SpeciesIntel.from_dict(
                     {"host_species": 75, "static": {"present": True},
                      "census": {"present": True},
                      "pairs": [{"main": 4, "sub": 15, "handler": "0x1",
                                 "measured": {"entered": 0}}]}))
    codes = {f.code: f for f in al.findings}
    assert codes["NEVER_ENTERED"].level == A.WARN, codes["NEVER_ENTERED"]
    assert "your call" in codes["NEVER_ENTERED"].message
    assert not al.errors, al.errors
    print("census            allow_unentered -> WARN, and says it is the author's call")


def test_an_absent_census_is_unknown_and_never_zero():
    """🔴 The distinction the whole intel layer is built around."""
    al = A.align(_m(), "charge", _intel())
    assert any(f.code == "UNMEASURED" for f in al.warnings), al.findings
    assert not [f for f in al.findings if f.code == "NEVER_ENTERED"]
    assert "UNKNOWN — not zero" in [f.message for f in al.findings
                                    if f.code == "UNMEASURED"][0]
    print("census            absent -> UNMEASURED, never NEVER_ENTERED")


def test_a_measured_pair_reports_its_dwell():
    al = A.align(_m(), "charge",
                 _intel(measured={"entered": 46, "dwell_ticks": 23.7,
                                  "move_per_tick": 1127.0}))
    f = [x for x in al.findings if x.code == "DWELL"]
    assert f and "46 time(s)" in f[0].message and "23.7" in f[0].message
    assert "1127 u/tick" in f[0].message, f[0].message
    print("census            a measured pair reports entries, dwell and travel")


# --------------------------------------------------------------------------- #
# effects
# --------------------------------------------------------------------------- #
def test_host_effect_bones_are_flagged_as_the_hosts_and_placed_on_your_rig():
    """⚠️ This is why the Zinogre threw snowballs out of Tigrex mouth bones."""
    al = A.align(_m(), "charge",
                 _intel(effects=[{"id": 60, "bone": 33, "frame": None}]), rig=RIG)
    f = [x for x in al.warnings if x.code == "EFFECT_BONES_ARE_THE_HOSTS"]
    assert f, al.findings
    assert "snowballs" in f[0].message
    assert "joint 33, 120 vertices, driven" in f[0].message, f[0].message
    assert any(x.code == "HOST_EFFECTS" for x in al.findings)
    print("effects           host bone numbers are named as the host's, and located")


def test_an_effect_bone_off_the_end_of_the_port_rig_is_an_error():
    al = A.align(_m(), "charge",
                 _intel(effects=[{"id": 60, "bone": 90, "frame": None}]), rig=RIG)
    assert [f for f in al.errors if f.code == "EFFECT_BONE_RANGE"], al.findings
    print("effects           a bone past the port rig -> ERROR (it reads off the array)")


def test_an_effect_on_a_joint_no_clip_drives_warns():
    al = A.align(_m(), "charge",
                 _intel(effects=[{"id": 60, "bone": 45, "frame": None}]), rig=RIG)
    f = [x for x in al.warnings if x.code == "EFFECT_BONE_UNDRIVEN"]
    assert f and "not follow the animal" in f[0].message, al.findings
    print("effects           an undriven anchor joint warns")


def test_a_framed_effect_becomes_a_marker_and_is_checked_against_the_clip():
    al = A.align(_m(), "charge",
                 _intel(effects=[{"id": 60, "bone": 33, "frame": 4},
                                 {"id": 61, "bone": 33, "frame": 900}]),
                 clip_frames=382, rig=RIG)
    fx = [mk for mk in al.markers if mk.kind == A.EFFECT]
    assert [mk.frame for mk in fx] == [4.0, 900.0], fx
    assert [mk.unreachable for mk in fx] == [False, True]
    assert any(f.code == "EFFECT_BEYOND_CLIP" for f in al.warnings)
    print("effects           a framed spawn is a marker, and one past the end warns")


def test_our_own_effects_are_drawn_apart_from_the_hosts_and_only_for_this_move():
    m = _m('\n[[effect]]\nmove = "charge"\nframe = 41\nid = 42\nbone = 33\n'
           '\n[moves.other]\nmain = 2\nsub = 9\nclip = "charge"\n'
           '\n[[effect]]\nmove = "other"\nframe = 7\nid = 43\nbone = 33\n')
    al = A.align(m, "charge", _intel(), rig=RIG)
    ours = [mk for mk in al.markers if mk.kind == A.OURS]
    assert [mk.frame for mk in ours] == [41.0], ours
    assert "ITS bone 33" in ours[0].detail, ours[0].detail
    print("effects           [[effect]] rows are OUR kind, and scoped to the move")


# --------------------------------------------------------------------------- #
# absent evidence
# --------------------------------------------------------------------------- #
def test_no_species_json_says_how_to_build_it_rather_than_failing():
    al = A.align(_m(), "charge", None)
    assert "tools/em_intel.py" in al.headline, al.headline
    assert any(f.code == "INTEL_ABSENT" for f in al.warnings)
    assert [mk.kind for mk in al.markers] == [A.IMPACT], al.markers
    print("absent            no intel -> a warning and the command, not an exception")


def test_a_pair_nothing_knows_about_says_exactly_that():
    al = A.align(_m('\n[moves.mystery]\nmain = 9\nsub = 99\nclip = "charge"\n'),
                 "mystery", _intel())
    assert "neither the overlay's jump tables nor the census" in al.headline
    print("absent            an unknown pair is named as unknown")


def test_align_all_covers_every_move_and_align_refuses_an_unknown_one():
    m = _m('\n[moves.second]\nmain = 2\nsub = 9\nclip = "charge"\n')
    got = A.align_all(m, _intel(event_frames=[60.0]), clip_frames={61: 382})
    assert [a.move for a in got] == ["charge", "second"], got
    assert got[0].frames == 382, got[0].frames
    try:
        A.align(m, "nope", None)
    except KeyError as e:
        assert "charge" in str(e), e
    else:
        raise AssertionError("an undeclared move was aligned")
    print("api               align_all covers every move; a bad name raises")


def test_the_marker_map_is_the_shape_the_timeline_draws():
    al = A.align(_m(), "charge", _intel(event_frames=[40.0, 60.0]))
    mm = al.marker_map()
    assert set(mm) == {40.0, 60.0, 41.0}, mm
    assert mm[41.0] == "impact", mm
    print("api               marker_map -> {frame: label}")


def test_port_rig_describes_a_joint_without_pretending_to_know_more():
    assert "does not exist" in RIG.describe(99)
    assert RIG.describe(33) == "joint 33, 120 vertices, driven"
    assert "NOT ANIMATED" in RIG.describe(45)
    assert "no geometry" in RIG.describe(2)
    print("api               PortRig.describe: exists / geometry / driven, nothing more")


# --------------------------------------------------------------------------- #
# writing the alignment down
# --------------------------------------------------------------------------- #
def test_binding_a_pair_writes_a_move_and_keeps_the_comments():
    import tempfile
    from mhfu_monster_editor import clips as C

    with tempfile.TemporaryDirectory() as d:
        path = os.path.join(d, "t.toml")
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("# a comment that must survive\n" + BASE.replace(
                '[moves.charge]\nmain = 2\nsub = 8\nclip = "charge"\n', ""))
        s = C.LabelSession(MF.load(path), {61: (382, False)}, "t.bin@abc")
        s.stage_move("charge", 2, 8, "charge")
        s.save()
        m = MF.load(path)
        assert (m.moves["charge"].main, m.moves["charge"].sub) == (2, 8)
        assert m.moves["charge"].clip == "charge"
        assert m.moves["charge"].latch == 1, "latch should keep its default"
        assert "# a comment that must survive" in open(path).read()
    print("bind              a pair + a clip becomes [moves.x], comments intact")


def test_binding_refuses_a_clip_the_manifest_has_not_named():
    from mhfu_monster_editor import clips as C

    s = C.LabelSession(_m(), {61: (382, False)})
    try:
        s.stage_move("x", 2, 8, "no_such_clip")
    except MF.ManifestError as e:
        assert "not named in this manifest" in str(e), e
    else:
        raise AssertionError("a move was bound to an undeclared clip")
    assert s.pending == 0
    print("bind              a move cannot point at a clip that is not declared")


# --------------------------------------------------------------------------- #
# the real overlay
# --------------------------------------------------------------------------- #
def test_the_real_em75_carries_the_frames_the_issue_quotes():
    from mhfu_monster_editor.intel import find_intel

    intel = find_intel(75, os.path.join(_ROOT, "species"))
    if intel is None:
        print("SKIP: no species/em75.json — python tools/em_intel.py --all")
        return
    # "113 of em75's 231 actions test fixed frame numbers" — the issue's figure is
    # `ends_on == "clip+cursor"`, which is the UNION of the two gate forms. 70 use the
    # plain `cursor >= F` and 114 the windowed one; neither alone is the number, which
    # is exactly why `window_frames` had to stop being a bare count.
    tested = [p for p in intel if p.fixed_event_frames]
    windowed = [p for p in intel if p.fixed_window_frames]
    both = [p for p in intel if p.tested_frames]
    assert len(both) >= 113, len(both)
    assert len(tested) == 70 and len(windowed) == 114, (len(tested), len(windowed))
    p = intel.pair(1, 13)
    # ⚠️ `fixed_event_frames` keeps one entry per CALL SITE — (1,13) tests 50 and 60
    # twice each — because how many times a handler asks is a real property. It is
    # `tested_frames` that dedupes, and that is what the timeline draws.
    assert p.fixed_event_frames == [40.0, 50.0, 50.0, 60.0, 60.0], p.event_frames
    assert 58.0 in p.fixed_window_frames, p.window_frames
    assert p.tested_frames == [-5.0, 40.0, 50.0, 58.0, 60.0], p.tested_frames

    m = _m()
    al = A.align_pair(m, 1, 13, intel, clip="charge", slot=61, clip_frames=382,
                      impact=41)
    assert "tests frames -5, 40, 50, 58, 60" in al.headline, al.headline
    assert "impact is at frame 41" in al.headline, al.headline
    print("em75              %d pairs name a fixed frame (%d plain, %d windowed); "
          "(1,13) -> %s" % (len(both), len(tested), len(windowed), al.headline))


def test_every_framed_effect_site_in_em75_is_unattributed_and_says_so():
    """The `id @ bone @ frame` numbers exist, but not on any pair: em75's 30 framed
    sites hang off a species-byte switch no pair handler calls."""
    from mhfu_monster_editor.intel import find_intel

    intel = find_intel(75, os.path.join(_ROOT, "species"))
    if intel is None:
        print("SKIP: no species/em75.json")
        return
    assert not [e for p in intel for e in p.effects if e.frame is not None], \
        "a pair-attributed recipe now carries a frame — the inspector can use it"
    fx = intel.framed_effects()
    assert len(fx) == 30, len(fx)
    assert all(e.frame is not None and e.bone is not None for e in fx)
    print("em75              %d framed effect sites, all species-wide, none on a pair"
          % len(fx))


def test_only_half_of_em75s_actions_name_a_clip_the_tigrex_pack_has():
    """🔴 The finding the host-reference view (#34) has to be honest about.

    `a1` IS the clip slot, so a pair's `a1` names the host's own animation for that
    action — except that half of them name slots the host's PAC does not populate. A
    view that quietly fell through to another a1, or showed the default pose without
    saying so, would display the wrong animation for the action.
    """
    from mhfu_monster_editor.clips import clip_table
    from mhfu_monster_editor.intel import find_intel

    pac = os.path.join(_ROOT, "workspace", "extracted", "data_files", "file_06185.bin")
    intel = find_intel(75, os.path.join(_ROOT, "species"))
    if intel is None or not os.path.exists(pac):
        print("SKIP: needs species/em75.json and the Tigrex PAC")
        return
    with open(pac, "rb") as fh:
        host = clip_table(fh.read())
    named = [p for p in intel if p.a1]
    ok = [p for p in named if all(a in host for a in p.a1)]
    none = [p for p in named if not any(a in host for a in p.a1)]
    assert len(named) > 200, len(named)
    assert 0.4 < len(ok) / len(named) < 0.6, (len(ok), len(named))
    assert len(ok) + len(none) > len(named) * 0.95, "the split is not clean"

    # the split is by SLOT NUMBER: the pack populates 64 slots in 1..98, and what the
    # handlers name above ~84 is mostly not there.
    missing = sorted({a for p in none for a in p.a1 if a not in host})
    assert min(missing) >= 55 and sum(1 for a in missing if a >= 84) > len(missing) * 0.7
    # and a computed a1 is far likelier to be one of the unresolvable ones
    rate = lambda g: sum(1 for p in g if p.a1_computed) / len(g)      # noqa: E731
    assert rate(none) > rate(ok) * 3, (rate(none), rate(ok))
    print("host a1           %d/%d pairs name a slot the Tigrex pack has; %d name none "
          "(computed-a1 rate %.0f%% vs %.0f%%)"
          % (len(ok), len(named), len(none), 100 * rate(none), 100 * rate(ok)))


def main() -> int:
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for fn in fns:
        try:
            fn()
        except AssertionError as e:
            failed += 1
            print("FAIL %s: %s" % (fn.__name__, e))
    print("\n%d tests, %d failed" % (len(fns), failed))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
