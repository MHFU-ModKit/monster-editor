"""The reader of `species/emNN.json`, and the one distinction it exists to protect.

Three states look alike in a JSON file and are three different findings:

    no census at all          -> `entered is None`, every consumer says UNKNOWN
    a census that never saw   -> `entered is None` too, but the FILE says a census ran
    a census that measured 0  -> `entered == 0` — the engine looked and never went
                                 there, 411 of 411 forced moves survived one tick

Only the last one may refuse a bind. Collapsing the first into the last would reject
every pair on every machine that has not run the probe — which is every machine today.
Collapsing the last into the first would let the editor bind the exact pair that wrecked
the first working Brute showcase. Both mistakes are one `or 0` away, so they are pinned
here rather than left to review.

No game data needed: every document below is synthetic.
"""
import json
import os
import sys
import tempfile

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mhfu_monster_editor import intel as I
from mhfu_monster_editor import manifest as MF
from mhfu_monster_editor import validate as V


def _pair(main, sub, **over):
    """One pair record in the joined on-disk shape."""
    d = dict(main=main, sub=sub, handler="0x09D2AB60", a1=[14, 15],
             a1_computed=False, ends_on="clip", event_frames=[], windows=0,
             clip_done_reads=1, budget_reads=0,
             budget=dict(gated=False, phase0_seeds=[], post_hook_owns=None),
             effects=[], measured=None,
             provenance={"handler": "static", "a1": "static", "ends_on": "static",
                         "entered": "absent"})
    d.update(over)
    return d


def _doc(pairs, census=None):
    return dict(
        schema=I.SCHEMA, host_species=75,
        overlay=dict(name="em75.ovl", file="file_06108.bin"),
        action_tick="0x09D36A08",
        main_states=[dict(main=0, dispatcher="0x09D33EA0", sub_states=34,
                          enumerated=True),
                     dict(main=5, dispatcher="0x09D35140", sub_states=None,
                          enumerated=False)],
        static=dict(present=True),
        census=census or dict(present=False, reason="no [state] lines in the log"),
        pairs=pairs, unattributed_effects=[])


# --------------------------------------------------------------------------- #
# the three-way distinction
# --------------------------------------------------------------------------- #
def test_no_census_means_entered_is_UNKNOWN_and_never_zero():
    """🔴 The whole point. `measured: null` must NOT arrive as `entered = 0`."""
    si = I.SpeciesIntel.from_dict(_doc([_pair(4, 15)]))
    p = si.pair(4, 15)
    assert p.entered is None, "an absent measurement read back as %r" % p.entered
    assert p.measured is False
    assert p.never_entered is False, "absent evidence is not a never-entered finding"
    assert si.has_census is False
    assert si.has_static is True


def test_a_census_that_measured_zero_is_a_finding_not_a_gap():
    si = I.SpeciesIntel.from_dict(_doc(
        [_pair(4, 15, measured=dict(entered=0, dwell_ticks=0.0, a1=[],
                                    move_per_tick=None, move_samples=0,
                                    note="0 of 1613 observed transitions"))],
        census=dict(present=True, transitions=1613, observed_pairs=37)))
    p = si.pair(4, 15)
    assert p.entered == 0 and p.measured is True and p.never_entered is True
    assert "1613" in p.note
    assert si.has_census is True


def test_a_pair_the_file_does_not_mention_is_None_not_a_zero():
    si = I.SpeciesIntel.from_dict(_doc([_pair(2, 8)]))
    assert si.pair(0, 3) is None


def test_the_static_note_and_the_census_note_do_not_get_spliced():
    """A pair can carry both. `note` is what the CENSUS said; the inline-dispatch
    explanation stays in `static_note`, so an error message never reads half
    measured and half inferred."""
    si = I.SpeciesIntel.from_dict(_doc([_pair(
        1, 12, handler=None, note="runs inline and calls no handler",
        measured=dict(entered=0, dwell_ticks=0.0, a1=[], move_per_tick=None,
                      move_samples=0, note="0 of 1613 observed transitions"))]))
    p = si.pair(1, 12)
    assert p.note == "0 of 1613 observed transitions"
    assert p.static_note == "runs inline and calls no handler"


# --------------------------------------------------------------------------- #
# the joined static fields
# --------------------------------------------------------------------------- #
def test_the_four_analysers_arrive_as_one_record():
    si = I.SpeciesIntel.from_dict(_doc([_pair(
        0, 19, handler="0x09D277C8", a1=[97], ends_on="budget",
        event_frames=[60.0, None], windows=3,
        budget=dict(gated=True, phase0_seeds=[], post_hook_owns=True),
        effects=[dict(id=24, bone=37, frame=56, site="0x09D40200", via="framed",
                      fn="0x09D40130"),
                 dict(id=79, bone=37, frame=None, site="0x09D27998", via="local",
                      fn="0x09D277C8")])]))
    p = si.pair(0, 19)
    assert p.handler == 0x09D277C8                       # em_moveset
    assert p.a1 == [97] and p.a1_static == [97]
    assert p.ends_on == "budget" and p.budget.post_hook_owns is True   # em_phase_map
    assert p.event_frames == [60.0, None] and p.fixed_event_frames == [60.0]
    assert p.ends_on_clip is False
    assert [str(e) for e in p.effects] == ["24@b37@f56", "79@b37"]     # em_effects
    assert si.pairs_with_effects() == [p] and si.budget_gated() == [p]


def test_an_unresolved_cursor_threshold_stays_None_rather_than_becoming_a_number():
    """em_phase_map prints `?` for a threshold loaded from data. Rounding that to 0
    would tell a port to put its impact on frame 0."""
    si = I.SpeciesIntel.from_dict(_doc([_pair(1, 3, event_frames=[None, None])]))
    assert si.pair(1, 3).fixed_event_frames == []


def test_a_measured_a1_wins_over_the_static_one_and_says_so():
    si = I.SpeciesIntel.from_dict(_doc(
        [_pair(2, 8, a1=[14, 15, 23],
               measured=dict(entered=46, dwell_ticks=23.7, a1=[15],
                             move_per_tick=1127.0, move_samples=40))],
        census=dict(present=True, transitions=1613, observed_pairs=37)))
    p = si.pair(2, 8)
    assert p.a1 == [15] and p.a1_static == [14, 15, 23]
    assert p.a1_provenance == I.MEASURED
    assert I.SpeciesIntel.from_dict(_doc([_pair(2, 8)])).pair(2, 8).a1_provenance \
        == I.STATIC


# --------------------------------------------------------------------------- #
# refusing a bind
# --------------------------------------------------------------------------- #
def _census_doc():
    return _doc([_pair(2, 8, measured=dict(entered=46, dwell_ticks=23.7, a1=[15],
                                           move_per_tick=1127.0, move_samples=40)),
                 _pair(4, 15, measured=dict(entered=0, dwell_ticks=0.0, a1=[],
                                            move_per_tick=None, move_samples=0)),
                 _pair(3, 6, measured=dict(entered=120, dwell_ticks=1.0, a1=[9],
                                           move_per_tick=649.0, move_samples=60))],
                census=dict(present=True, transitions=1613, observed_pairs=37))


def test_a_never_entered_pair_is_refused_and_an_override_is_recorded():
    si = I.SpeciesIntel.from_dict(_census_doc())
    b = si.bindable(4, 15)
    assert not b.ok and b.code == I.BIND_NEVER_ENTERED and not b.overridden
    b = si.bindable(4, 15, override=True)
    assert b.ok and b.overridden and b.code == I.BIND_NEVER_ENTERED


def test_a_pair_the_engine_uses_binds_cleanly():
    b = I.SpeciesIntel.from_dict(_census_doc()).bindable(2, 8)
    assert b.ok and b.code == I.BIND_OK and not b.unverified


def test_a_short_dwell_binds_but_is_flagged_unverified():
    b = I.SpeciesIntel.from_dict(_census_doc()).bindable(3, 6)
    assert b.ok and b.code == I.BIND_SHORT_DWELL and b.unverified


def test_without_a_census_a_bind_is_allowed_but_never_reported_as_verified():
    """The census is absent on every machine that has not run the probe. Refusing
    there would make the editor unusable; claiming OK would be a lie."""
    b = I.SpeciesIntel.from_dict(_doc([_pair(4, 15)])).bindable(4, 15)
    assert b.ok and b.unverified and b.code == I.BIND_UNMEASURED
    assert "UNKNOWN" in b.reason


def test_a_sub_state_the_dispatcher_does_not_have_is_refused_from_STATIC_alone():
    """This one needs no census at all: main 0 has 34 cases, so 99 is not an action."""
    b = I.SpeciesIntel.from_dict(_doc([_pair(0, 19)])).bindable(0, 99)
    assert not b.ok and b.code == I.BIND_NO_HANDLER and "34" in b.reason


def test_a_main_state_the_extractor_could_not_enumerate_is_not_called_missing():
    """Mains 5/6/7 of em75 have no sub_state jump table. "We cannot see it" is not
    the same claim as "it does not exist"."""
    b = I.SpeciesIntel.from_dict(_doc([_pair(0, 19)])).bindable(5, 2)
    assert b.ok and b.unverified and b.code == I.BIND_UNKNOWN_PAIR


def test_an_inline_case_binds_unverified_rather_than_being_refused():
    si = I.SpeciesIntel.from_dict(_doc([_pair(1, 12, handler=None)]))
    b = si.bindable(1, 12)
    assert b.ok and b.unverified and b.code == I.BIND_NO_HANDLER


# --------------------------------------------------------------------------- #
# compatibility with what issue #2 wrote
# --------------------------------------------------------------------------- #
def test_the_provisional_flat_shape_still_reads():
    """`validate.JsonActionIntel` documented `{"pairs":[{main,sub,entered,...}]}` and
    called itself "the single class #4 replaces". Replacing it must not orphan a file
    somebody already wrote by hand."""
    si = I.SpeciesIntel.from_dict({
        "host_species": 75,
        "pairs": [{"main": 2, "sub": 8, "entered": 46, "dwell_ticks": 23.7,
                   "a1": [15]}]})
    assert si.host_species == 75 and len(si) == 1
    p = si.pair(2, 8)
    assert p.entered == 46 and p.dwell_ticks == 23.7 and p.a1 == [15]
    assert si.has_census is True, "a flat file IS a census — it has nothing else"
    assert si.has_static is False, "and it carries no static intel"
    assert si.pair(4, 15) is None


def test_the_reader_satisfies_the_protocol_validate_was_written_against():
    si = I.SpeciesIntel.from_dict(_doc([_pair(2, 8)]))
    assert isinstance(si, I.ActionIntel)
    assert isinstance(si, V.ActionIntel), "validate re-exports the same protocol"
    assert V.JsonActionIntel is I.SpeciesIntel


def test_find_intel_reads_a_written_file_and_returns_None_without_one():
    with tempfile.TemporaryDirectory() as td:
        assert I.find_intel(75, root=td) is None
        with open(os.path.join(td, "em75.json"), "w") as fh:
            json.dump(_doc([_pair(2, 8)]), fh)
        si = I.find_intel(75, root=td)
        assert si is not None and si.host_species == 75 and len(si) == 1
        assert I.find_intel(17, root=td) is None
    assert V.find_intel is I.find_intel


# --------------------------------------------------------------------------- #
# what the validator does with it
# --------------------------------------------------------------------------- #
BASE = """
[port]
name = "t"
host_species = 75
pac = "t.bin"

[source]
em_id = 58
model = 5248

[build]
source_skeleton = true
skin = "source"

[clips.c]
slot = 61
"""
MOVE = "\n[moves.m]\nmain = %d\nsub = %d\nclip = \"c\"\n"


def _codes(iss):
    return {i.code for i in iss}


def test_a_census_free_file_warns_INTEL_ABSENT_and_does_not_reject_the_pair():
    si = I.SpeciesIntel.from_dict(_doc([_pair(4, 15)]))
    iss = V.validate(MF.loads(BASE + MOVE % (4, 15)), pac=None, intel=si)
    assert "INTEL_ABSENT" in _codes(iss), iss
    assert "MOVE_PAIR_NEVER_ENTERED" not in _codes(iss)
    assert not [i for i in iss if i.level == V.ERROR], iss


def test_the_same_pair_is_an_ERROR_once_a_census_has_measured_it():
    si = I.SpeciesIntel.from_dict(_census_doc())
    iss = V.validate(MF.loads(BASE + MOVE % (4, 15)), pac=None, intel=si)
    bad = [i for i in iss if i.code == "MOVE_PAIR_NEVER_ENTERED"]
    assert len(bad) == 1 and bad[0].level == V.ERROR, iss
    assert "INTEL_ABSENT" not in _codes(iss)


def test_allow_unentered_is_the_explicit_override_and_it_is_still_reported():
    si = I.SpeciesIntel.from_dict(_census_doc())
    body = BASE + MOVE % (4, 15) + "allow_unentered = true\n"
    iss = V.validate(MF.loads(body), pac=None, intel=si)
    bad = [i for i in iss if i.code == "MOVE_PAIR_NEVER_ENTERED"]
    assert len(bad) == 1 and bad[0].level == V.WARNING, iss
    assert "allow_unentered" in bad[0].message
    assert not [i for i in iss if i.level == V.ERROR], iss


def test_a_sub_state_outside_the_dispatcher_is_an_error_with_no_census_at_all():
    si = I.SpeciesIntel.from_dict(_doc([_pair(0, 19)]))
    iss = V.validate(MF.loads(BASE + MOVE % (0, 99)), pac=None, intel=si)
    bad = [i for i in iss if i.code == "MOVE_PAIR_NO_HANDLER"]
    assert len(bad) == 1 and bad[0].level == V.ERROR, iss


def test_a_budget_gated_pair_warns_that_a_longer_clip_is_truncated():
    si = I.SpeciesIntel.from_dict(_doc([_pair(
        0, 19, ends_on="budget",
        budget=dict(gated=True, phase0_seeds=[150], post_hook_owns=False))]))
    iss = V.validate(MF.loads(BASE + MOVE % (0, 19)), pac=None, intel=si)
    got = [i for i in iss if i.code == "MOVE_PAIR_BUDGET_GATED"]
    assert len(got) == 1 and got[0].level == V.WARNING, iss
    assert "150" in got[0].message and "slot-29" in got[0].message


def test_intel_for_the_wrong_species_still_stops_before_it_guesses():
    d = _doc([_pair(2, 8)])
    d["host_species"] = 76
    iss = V.validate(MF.loads(BASE + MOVE % (2, 8)), pac=None,
                     intel=I.SpeciesIntel.from_dict(d))
    assert "INTEL_WRONG_SPECIES" in _codes(iss), iss
    assert "MOVE_PAIR_UNOBSERVED" not in _codes(iss)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
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
