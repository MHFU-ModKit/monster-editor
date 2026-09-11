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


def test_available_and_survey_list_every_overlay_on_this_machine():
    """The "which host should this port ride?" question — 17 overlays, one shape.

    A ported monster has no AI of its own; the engine loads ONE MHFU species overlay
    for it and `port.host_species` picks which. So the chooser has to be able to see
    all of them, not just the Tigrex.
    """
    import os

    root = os.path.join(_ROOT, "species")
    ids = I.available(root)
    if not ids:
        print("SKIP: no species/*.json — python tools/em_intel.py --all")
        return
    assert all(isinstance(i, int) for i in ids) and ids == sorted(ids), ids
    hosts = I.survey_hosts(root)
    assert [h.species for h in hosts] == ids, [h.species for h in hosts]
    for h in hosts:
        assert h.handled <= h.pairs and h.timed <= h.pairs, h
        assert h.free_timing == h.pairs - h.timed
        assert 0 <= h.opaque_mains <= 8, h          # every overlay has exactly 8 mains
    big = max(hosts, key=lambda h: h.pairs)
    assert big.species == 75, big.species           # the Tigrex is the richest of them
    print("hosts             %d overlays; em75 is the largest at %d pairs (%d timed, "
          "%d budget-gated)" % (len(hosts), big.pairs, big.timed, big.budget))



# --------------------------------------------------------------------------- #
# the part system (#10)
# --------------------------------------------------------------------------- #
def test_part_intel_is_never_none_so_a_caller_never_guards_it():
    si = I.SpeciesIntel.from_dict({"host_species": 75, "pairs": []})
    assert si.parts is not None
    assert si.parts.present is False and si.parts.spheres() == []
    assert si.parts.has_grid is False


def test_a_sphere_keeps_part_and_row_apart():
    s = I.HitSphere.from_dict({"bone": 12, "part": 6, "hitzone_row": 5,
                               "radius": 90.0, "a": [-100.0, 0.0, 0.0]})
    assert (s.part, s.hitzone_row) == (6, 5)
    assert s.is_capsule is False and s.b is None


def test_a_capsule_carries_its_far_end():
    s = I.HitSphere.from_dict({"bone": 6, "part": 4, "hitzone_row": 5, "radius": 65.0,
                               "shape": "capsule", "a": [35.0, 0, 0],
                               "b": [330.0, 0, 0]})
    assert s.is_capsule and s.b == (330.0, 0.0, 0.0)


def test_only_the_sets_that_name_parts_count_as_hurtboxes():
    """A part-less set is the thing a cold boot disproved: zeroing those four radii
    did not stop damage. It must not be drawn as somewhere he can be hit."""
    pt = I.PartIntel.from_dict({
        "present": True,
        "sets": [{"va": "0x09D58CD0", "kind": "hurtbox",
                  "spheres": [{"bone": 2, "part": 1, "hitzone_row": 2, "radius": 97.0}]},
                 {"va": "0x09D5EB48", "kind": "volume",
                  "spheres": [{"bone": 10, "part": 0, "hitzone_row": 0,
                               "radius": 150.0}]}],
    })
    assert len(pt.sets) == 2 and len(pt.hurtboxes) == 1
    assert [s.bone for s in pt.spheres()] == [2]


def test_a_part_may_use_more_than_one_hitzone_row_and_says_so():
    """Tigrex part 6 uses rows 3 and 5 — its spheres take DIFFERENT percentages.
    Averaging them, or reporting the first, would invent a number."""
    si = I.find_intel(75)
    if si is None or not si.parts.present:
        return
    # ⚠️ corrected 2026-09-11: "part 6 uses rows 3 AND 5" was an artefact of merging
    # all four em75 sets — set 0 (the one species 75 walks) has the wing on row 5
    # only; row 3 is species 81's and 88's. The real Tigrex example is the HEAD:
    # part 1's spheres sit on rows 1 and 2.
    if si.parts.active is not None:
        assert si.parts.rows_of_part(6) == [5], si.parts.rows_of_part(6)
        assert si.parts.rows_of_part(1) == [1, 2], si.parts.rows_of_part(1)
    else:
        assert si.parts.rows_of_part(6) == [3, 5], si.parts.rows_of_part(6)


def test_the_inferred_column_names_are_flagged_as_inferred():
    """Six of the ten column names were not read out of the game. A panel that
    renders `thunder` the way it renders `cut` is overclaiming, and it can only
    avoid that if the data marks them."""
    si = I.find_intel(75)
    if si is None or not si.parts.present:
        return
    inferred = si.parts.inferred_columns()
    assert "thunder" in inferred and "ko" in inferred
    assert "cut" not in inferred and "impact" not in inferred


def test_the_tigrex_grid_comes_through_the_editor_layer_intact():
    si = I.find_intel(75)
    if si is None or not si.parts.present or not si.parts.has_grid:
        return
    pt = si.parts
    assert pt.n_states == 2
    assert len(pt.states[0].rows) == I.HITZONE_ROWS
    assert all(len(r) == len(I.HITZONE_COLUMNS) for r in pt.states[0].rows)
    assert pt.states[0].value(0, "cut") == 75
    assert pt.states[0].value(0, "ko") == 110
    assert pt.grid_note, "the shared-grid warning did not survive the join"


def test_every_part_maps_to_bones_on_the_rig():
    si = I.find_intel(75)
    if si is None or not si.parts.present:
        return
    assert si.parts.parts() == list(range(8)), si.parts.parts()
    for part in si.parts.parts():
        assert si.parts.bones_of_part(part), "part %d has no bones" % part



# --------------------------------------------------------------------------- #
# the attack side (#33) — the reader, and the em75 anchors that were measured
# --------------------------------------------------------------------------- #
_ATTACKS = {
    "present": True, "spawner": "0x09B661E8", "join": "measured",
    "id_offsets": {"75": 0, "76": 33, "88": 70},
    "field_provenance": {"power": "measured", "kind": "shape only"},
    "tables": [
        {"handle": "0x09D61250", "records": "0x09D60848", "volume_table": "0x09D60768",
         "primary": True, "rigged": True,
         "sets": [
             {"index": 0, "va": "0x1000", "count": 1, "rigged": True,
              "spheres": [{"bone": 35, "radius": 180.0, "a": [0, 0, 0]}]},
             {"index": 1, "va": "0x1030", "count": 2, "rigged": False,
              "spheres": [{"bone": 126, "shape": "capsule", "radius": 150.0,
                           "a": [0, 0, 0], "b": [0, 0, 1]}]},
             {"index": 2, "va": "0x1090", "count": 3,
              "spheres": [{"bone": 10, "radius": 150.0}, {"bone": 125, "shape": "capsule",
                                                          "radius": 0.0, "row_part": [1, 0]},
                          {"bone": 43, "radius": 120.0}]}],
         "attacks": [
             {"id": 0, "va": "0x2000", "power": 0, "element": "0x00", "volume": 0,
              "raw": "00" * 24},
             {"id": 1, "va": "0x2018", "power": 30, "element": "0x21", "volume": 0,
              "raw": "001e" + "00" * 22},
             {"id": 6, "va": "0x2090", "power": 64, "element": "0x21", "volume": 2,
              "kind": 2, "tag": "0x28", "value_14": 25, "raw": "0063" + "00" * 22},
             {"id": 31, "va": "0x2300", "power": 30, "element": "0x01", "volume": 2,
              "raw": "001e" + "00" * 22}]},
        {"handle": "0x3000", "records": "0x3010", "volume_table": "0x3008",
         "primary": False, "rigged": False, "sets": [], "attacks": []}],
}


def test_attack_intel_reads_the_join_and_keeps_record_zero_out():
    at = I.AttackIntel.from_dict(_ATTACKS)
    assert at.present and at.spawner == 0x09B661E8 and at.join == "measured"
    assert at.primary is at.tables[0] and len(at.tables) == 2
    assert [a.id for a in at.attacks] == [1, 6, 31], "record 0 is blank, not an attack"
    a6 = at.attack(6)
    assert a6.power == 64 and a6.element == 0x21 and a6.volume == 2 and a6.value_14 == 25
    assert at.attack(0).is_blank and at.attack(99) is None


def test_attack_sets_resolve_and_the_marker_bones_are_a_coordinate_space():
    at = I.AttackIntel.from_dict(_ATTACKS)
    s2 = at.set(2)
    assert s2.capacity == 3 and s2.bones == [10, 43], "125 is a joiner, not a joint"
    assert s2.rigged, "a set with real joints is rigged even with a marker in it"
    assert not at.set(1).rigged, "one capsule on bone 126 hangs on the node, not a rig"
    assert at.set(1).spheres[0].is_capsule
    assert at.capacity(2) == 3 and at.capacity(7) is None
    # row_part is carried for the round trip; a HitSphere reads it as nothing special
    assert s2.spheres[1].bone == I.ATTACK_BONE_JOINER


def test_the_move_to_set_join_goes_through_the_records():
    at = I.AttackIntel.from_dict(_ATTACKS)
    assert at.sets_for([6, 31]) == [2]
    assert [a.id for a in at.records_for([6, 31])] == [6, 31]
    assert [a.id for a in at.attacks_using(2)] == [6, 31]
    assert at.attacks_using(0) and at.attacks_using(0)[0].id == 1
    assert at.sets_for([0]) == [], "a blank record hits with nothing"


def test_a_species_the_overlay_was_not_read_for_gets_no_offset_not_zero():
    at = I.AttackIntel.from_dict(_ATTACKS)
    assert at.id_offset(75) == 0 and at.id_offset(76) == 33 and at.id_offset(88) == 70
    assert at.id_offset(81) is None
    assert at.records_for([6], entity_species=81) == [], "no offset -> no records"
    # species 76 slices the same table +33: literal 1 -> record 34 (absent here)
    assert at.records_for([1], entity_species=76) == []
    assert at.records_for([1], entity_species=75)[0].id == 1


def test_an_absent_attacks_block_is_present_false_with_the_reason():
    at = I.AttackIntel.from_dict(None)
    assert not at.present and at.reason and at.primary is None
    assert at.sets == [] and at.attacks == [] and at.capacity(0) is None
    at2 = I.AttackIntel.from_dict({"present": False, "reason": "never calls the setter",
                                   "spawner": None})
    assert at2.reason == "never calls the setter"


def test_pair_intel_carries_the_handler_literals():
    p = I.PairIntel.from_dict({"main": 1, "sub": 4, "handler": "0x09D28738",
                               "attack_ids": [6, 31], "attack_sites": 7,
                               "attack_sites_computed": 0, "measured": None})
    assert p.attack_ids == [6, 31] and p.attack_sites == 7
    q = I.PairIntel.from_dict({"main": 0, "sub": 1, "measured": None})
    assert q.attack_ids == [] and q.attack_sites == 0


def test_species_intel_reverse_joins_pairs_to_a_set():
    si = I.SpeciesIntel.from_dict({
        "host_species": 75, "main_states": [], "attacks": _ATTACKS,
        "pairs": [{"main": 1, "sub": 4, "handler": "0x1", "attack_ids": [6, 31],
                   "measured": None},
                  {"main": 3, "sub": 9, "handler": "0x2", "attack_ids": [1],
                   "measured": None},
                  {"main": 0, "sub": 0, "handler": "0x3", "measured": None}]})
    assert [(p.main, p.sub) for p in si.pairs_hitting_with(2)] == [(1, 4)]
    assert [(p.main, p.sub) for p in si.pairs_hitting_with(0)] == [(3, 9)]
    assert si.pairs_hitting_with(1) == []


_CHAIN_DOC = {
    "host_species": 75, "main_states": [],
    "chain": {"hubs": [[0, 1], [0, 2]], "enter_action": "0x09D3D608"},
    "pairs": [
        {"main": 1, "sub": 4, "handler": "0x1", "measured": None,
         "next": [{"site": "0x10", "kind": "enter", "main": 0, "id": 3, "mode": 0,
                   "via": ["0x09D26158"], "to": [[0, 3]],
                   "guards": ["phase==3", "!collided", "budget spent"], "alts": []},
                  {"site": "0x11", "kind": "enter", "main": 0, "id": 6, "mode": 1,
                   "via": [], "to": [[0, 6]], "guards": ["phase==3", "collided"]}],
         "prev": [], "provenance": {"next": "static"}},
        {"main": 0, "sub": 3, "handler": "0x2", "measured": None,
         "next": [{"site": "0x20", "kind": "enter", "main": 0, "id": 1, "mode": 0,
                   "via": [], "to": [[0, 1], [0, 2]], "guards": ["phase==1"]}],
         "prev": [[1, 4]]},
        {"main": 0, "sub": 6, "handler": "0x3", "measured": None, "next": [],
         "prev": [[1, 4]]},
        {"main": 0, "sub": 1, "handler": "0x4", "measured": None, "next": [], "prev": [[0, 3]]},
        {"main": 0, "sub": 2, "handler": "0x5", "measured": None, "next": [], "prev": [[0, 3]]},
        {"main": 9, "sub": 9, "handler": "0x6", "measured": None},
    ]}


def test_edges_read_back_with_their_reason_phase_and_route():
    si = I.SpeciesIntel.from_dict(_CHAIN_DOC)
    assert si.has_chain and si.hubs == [(0, 1), (0, 2)]
    p = si.pair(1, 4)
    assert p.successors == [(0, 3), (0, 6)] and p.ends_itself is True
    e = p.next[0]
    assert e.to == ((0, 3),) and e.phase == 3 and e.raw_reason == "!collided & budget spent"
    assert e.via == (0x09D26158,) and e.mode == 0 and e.site == 0x10
    # a handled pair with an EMPTY next never ends itself; one without the field
    # (an old file) says unknown, not False
    assert si.pair(0, 6).ends_itself is False
    assert si.pair(9, 9).ends_itself is None and si.pair(9, 9).next is None
    assert [(q.main, q.sub) for q in si.predecessors(0, 3)] == [(1, 4)]
    assert si.successors(0, 3)[0].to == ((0, 1), (0, 2))


def test_pinned_cells_are_named_and_unpinned_ones_stay_hex():
    d = I.describe_guard
    assert d("+0x280!=0") == "reaction pending" and d("+0x280==0") == "no reaction pending"
    assert d("+0x324==1002") == "playing a1 2" and d("+0x324!=1017") == "not playing a1 17"
    assert d("+0x414<=0") == "frame budget spent" and d("budget spent") == "run budget spent"
    assert d("+0xBE==0") == "clip done" and d("+0x637==1") == "run budget armed"
    assert d("+0x29A==99") == "section==99"
    assert d("+0x6DB!=0") == "+0x6DB!=0"          # not pinned: stays a fact
    assert d("phase==3") == "phase==3" and d("!collided") == "!collided"
    si = I.SpeciesIntel.from_dict(_CHAIN_DOC)
    e = si.pair(1, 4).next[0]
    assert e.reason == "!collided & run budget spent"
    assert e.raw_reason == "!collided & budget spent"
    assert str(e) == "(0,3)  [phase==3 & !collided & run budget spent]"


def test_chain_from_walks_to_the_hubs_and_stops_there():
    si = I.SpeciesIntel.from_dict(_CHAIN_DOC)
    walk = [(q.main, q.sub) for q in si.chain_from(1, 4)]
    assert walk == [(1, 4), (0, 3), (0, 6), (0, 1), (0, 2)], walk
    # a hub as the start is expanded; as a terminal it is not
    assert [(q.main, q.sub) for q in si.chain_from(0, 1)] == [(0, 1)]
    assert [(q.main, q.sub) for q in si.entries()] == [(1, 4)]
    assert si.chain_from(7, 7) == []


def test_a_file_without_the_chain_block_has_no_chain():
    si = I.SpeciesIntel.from_dict({"host_species": 75, "main_states": [],
                                   "pairs": [{"main": 1, "sub": 4, "measured": None}]})
    assert not si.has_chain and si.hubs == [] and si.successors(1, 4) == []
    assert si.chain_from(1, 4) == [si.pair(1, 4)]


def test_em75_the_charge_pair_hits_with_set_two_as_measured_live():
    """(1,4) is the Tigrex charge measured at -72 HP; the live replacement of
    set 2 moved the hit 645 -> 152 -> 1381 units. The static join has to land
    on exactly that set or the whole editor would be pointing at the wrong table."""
    si = I.find_intel(75)
    if si is None or not si.attacks.present:
        return
    at = si.attacks
    assert at.join == "measured" and at.spawner == 0x09B661E8
    p = si.pair(1, 4)
    assert p is not None and 6 in p.attack_ids, p.attack_ids
    assert at.sets_for(p.attack_ids) == [2]
    assert at.attack(6).power == 64 and at.attack(6).volume == 2
    s2 = at.set(2)
    assert s2.capacity == 10 and s2.bones == [2, 4, 10, 18, 34, 41, 42, 43]
    assert at.id_offsets == {75: 0, 76: 33, 88: 70}
    # the four extras are the un-rigged projectile tables
    assert sum(1 for t in at.tables if not t.primary) == 4
    assert all(not t.rigged for t in at.tables if not t.primary and t.sets
               and len(t.attacks) > 1)

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
