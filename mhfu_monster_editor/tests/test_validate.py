"""The validator, on the three traps the issue names — plus what it must NOT do.

Two of these checks want a built PAC and one wants an action census, and neither is
guaranteed to exist. The clip/bone checks are pinned here against a stubbed clip table
so they run on any machine; `test_ports_build.py` re-runs them against the real thing.

The rule the "degrade gracefully" tests protect: **absent evidence warns, it never
passes silently and it never fails**. A validator that errors because it could not find
the census will be switched off, and a validator that says OK because it could not find
the census is worse than not having one.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)

from mhfu_monster_editor import manifest as MF
from mhfu_monster_editor import validate as V
from mhfu_monster_editor import intel as INTEL

BASE = """
[port]
name = "test"
host_species = 75
pac = "test.bin"

[source]
em_id = 40
model = 5339

[build]
source_skeleton = true
skin = "source"
"""

# what a built PAC's clip table looks like: {slot: (last_keyframe, loop)}.
# Slot 1 is the idle; the porter fills unclaimed host slots with a COPY of it.
FAKE_CLIPS = {1: (180, 1), 61: (382, 0), 69: (180, 1), 30: (392, 1)}
FAKE_BONES = 47


def _issues(text, **kw):
    """Validate with the PAC readers stubbed, so no game data is needed."""
    real_clip, real_bone = V.clip_table, V.bone_count
    V.clip_table = lambda blob: dict(FAKE_CLIPS)
    V.bone_count = lambda blob: FAKE_BONES
    try:
        return V.validate(MF.loads(text), pac=b"stub", **kw)
    finally:
        V.clip_table, V.bone_count = real_clip, real_bone


def _codes(issues):
    return {i.code for i in issues}


# --------------------------------------------------------------------------- #
# trap 1 — a clip slot that is not in the built PAC
# --------------------------------------------------------------------------- #
def test_a_clip_slot_absent_from_the_build_is_an_error():
    iss = _issues(BASE + "\n[clips.trapped]\nslot = 82\n")
    bad = [i for i in iss if i.code == "CLIP_SLOT_MISSING"]
    assert len(bad) == 1, iss
    assert bad[0].level == V.ERROR
    assert bad[0].where == "clips.trapped"


def test_a_clip_slot_present_in_the_build_passes():
    assert "CLIP_SLOT_MISSING" not in _codes(_issues(BASE + "\n[clips.c]\nslot = 61\n"))


def test_a_filler_slot_warns_because_it_plays_idle():
    """Slot 69 holds a copy of the idle clip. Forcing it is a SUCCESSFUL override onto
    nothing, and on screen that is identical to the override failing."""
    iss = _issues(BASE + "\n[clips.break_free]\nslot = 69\n")
    assert "CLIP_IS_FILLER" in _codes(iss), iss
    assert all(i.level == V.WARNING for i in iss if i.code == "CLIP_IS_FILLER")


def test_the_idle_slot_itself_is_not_flagged_as_filler():
    assert "CLIP_IS_FILLER" not in _codes(_issues(BASE + "\n[clips.idle]\nslot = 1\n"))


def test_a_frame_fingerprint_that_does_not_match_the_build_is_an_error():
    """Clips are stored UNPADDED at their authored length, so `frames` is an identity,
    not an approximation. A mismatch means the slot holds a different clip."""
    iss = _issues(BASE + "\n[clips.c]\nslot = 61\nframes = 408\n")
    assert "CLIP_FRAMES_MISMATCH" in _codes(iss), iss
    iss = _issues(BASE + "\n[clips.c]\nslot = 61\nframes = 382\nloop = true\n")
    assert "CLIP_LOOP_MISMATCH" in _codes(iss), iss
    iss = _issues(BASE + "\n[clips.c]\nslot = 61\nframes = 382\nloop = false\n")
    assert not _codes(iss) & {"CLIP_FRAMES_MISMATCH", "CLIP_LOOP_MISMATCH"}, iss


# --------------------------------------------------------------------------- #
# trap 2 — a (main,sub) pair the census says is never entered
# --------------------------------------------------------------------------- #
def _intel(**over):
    pairs = [V.PairIntel(2, 8, entered=46, dwell_ticks=23.7, a1=[15]),
             V.PairIntel(4, 15, entered=0),
             V.PairIntel(3, 6, entered=120, dwell_ticks=1.0)]
    return V.JsonActionIntel(over.get("species", 75), pairs)


MOVE = "\n[clips.c]\nslot = 61\n\n[moves.m]\nmain = %d\nsub = %d\nclip = \"c\"\n"


def test_a_never_entered_pair_is_rejected_when_the_census_is_there():
    iss = _issues(BASE + MOVE % (4, 15), intel=_intel())
    bad = [i for i in iss if i.code == "MOVE_PAIR_NEVER_ENTERED"]
    assert len(bad) == 1 and bad[0].level == V.ERROR, iss


def test_a_pair_the_engine_dwells_in_passes():
    iss = _issues(BASE + MOVE % (2, 8), intel=_intel())
    assert not [i for i in iss if i.level == V.ERROR], iss


def test_a_short_dwell_pair_warns_rather_than_fails():
    """(3,6) closes 649 units/tick when the ENGINE picks it and bounces out in one to
    seven ticks when forced from out of range. That is a judgement call, not a fault."""
    iss = _issues(BASE + MOVE % (3, 6), intel=_intel())
    assert "MOVE_PAIR_SHORT_DWELL" in _codes(iss), iss
    assert not [i for i in iss if i.level == V.ERROR], iss


def test_a_pair_the_census_never_saw_warns_it_is_UNOBSERVED_not_never_entered():
    """Absent from the sample is not the same finding as `entered == 0`."""
    iss = _issues(BASE + MOVE % (0, 7), intel=_intel())
    assert "MOVE_PAIR_UNOBSERVED" in _codes(iss), iss
    assert not [i for i in iss if i.level == V.ERROR], iss


def test_intel_for_the_wrong_host_species_is_refused():
    """(main,sub) is dispatched by the HOST overlay, so a census from another species
    says nothing at all — using it would be worse than having none."""
    iss = _issues(BASE + MOVE % (2, 8), intel=_intel(species=76))
    assert "INTEL_WRONG_SPECIES" in _codes(iss), iss
    assert "MOVE_PAIR_UNOBSERVED" not in _codes(iss), "it must stop, not guess"


# --------------------------------------------------------------------------- #
# trap 3 — a hurtbox bone out of range for the shipped skeleton
# --------------------------------------------------------------------------- #
def test_a_hurtbox_bone_past_the_end_of_the_rig_is_an_error():
    iss = _issues(BASE + "\n[[hurtbox]]\nbone = 47\nradius = 150\n")
    assert "HURTBOX_BONE_RANGE" in _codes(iss), iss
    iss = _issues(BASE + "\n[[hurtbox]]\nbone = -1\nradius = 150\n")
    assert "HURTBOX_BONE_RANGE" in _codes(iss), iss
    iss = _issues(BASE + "\n[[hurtbox]]\nbone = 46\nradius = 150\n")
    assert "HURTBOX_BONE_RANGE" not in _codes(iss), iss


def test_a_hurtbox_with_no_radius_is_not_a_collision_sphere():
    iss = _issues(BASE + "\n[[hurtbox]]\nbone = 10\nradius = 0\n")
    assert "HURTBOX_RADIUS" in _codes(iss), iss


def test_an_effect_bone_is_range_checked_too():
    """spawn_effect reads that bone's LIVE world position every call."""
    body = BASE + "\n[clips.c]\nslot = 61\n\n[moves.m]\nmain = 2\nsub = 8\nclip = \"c\"\n"
    iss = _issues(body + "\n[[effect]]\nmove = \"m\"\nframe = 4\nid = 42\nbone = 90\n")
    assert "EFFECT_BONE_RANGE" in _codes(iss), iss


# --------------------------------------------------------------------------- #
# degrading gracefully — the seam for issue #4
# --------------------------------------------------------------------------- #
def test_no_census_warns_loudly_and_passes():
    iss = _issues(BASE + MOVE % (4, 15))            # intel omitted
    assert "INTEL_ABSENT" in _codes(iss), iss
    assert not [i for i in iss if i.level == V.ERROR], iss
    assert "em_state_census" in next(i.message for i in iss if i.code == "INTEL_ABSENT")


def test_no_pac_warns_and_does_not_pretend_the_clips_were_checked():
    m = MF.loads(BASE + "\n[clips.trapped]\nslot = 82\n")
    iss = V.validate(m, pac=None)
    assert "PAC_ABSENT" in _codes(iss), iss
    assert "CLIP_SLOT_MISSING" not in _codes(iss)
    assert not [i for i in iss if i.level == V.ERROR], iss


def test_a_manifest_with_nothing_to_check_is_silent():
    """No clips, no moves, no hurtboxes -> no evidence needed, so no nagging."""
    assert V.validate(MF.loads(BASE), pac=None) == []


def test_find_intel_returns_none_when_issue_4_has_not_run():
    assert V.find_intel(75, root=os.path.join(_ROOT, "no-such-dir")) is None


def test_the_json_intel_reader_speaks_the_documented_shape():
    ai = V.JsonActionIntel.from_dict({
        "host_species": 75,
        "pairs": [{"main": 2, "sub": 8, "entered": 46, "dwell_ticks": 23.7,
                   "a1": [15]}]})
    assert ai.host_species == 75 and len(ai) == 1
    p = ai.pair(2, 8)
    assert p.entered == 46 and p.dwell_ticks == 23.7 and p.a1 == [15]
    assert ai.pair(4, 15) is None
    assert isinstance(ai, V.ActionIntel), "the seam is a runtime-checkable protocol"


# --------------------------------------------------------------------------- #
# the derived-value sanity checks
# --------------------------------------------------------------------------- #
def test_a_wrong_fid_warns_rather_than_failing():
    iss = V.validate(MF.loads(BASE.replace('pac = "test.bin"',
                                           'pac = "test.bin"\nfid = 1234')), pac=None)
    assert "FID_UNEXPECTED" in _codes(iss), iss
    assert not [i for i in iss if i.level == V.ERROR], iss


def test_source_skin_without_the_source_rig_warns():
    body = BASE.replace("source_skeleton = true", "source_skeleton = false")
    assert "SKIN_SOURCE_WITHOUT_RIG" in _codes(V.validate(MF.loads(body), pac=None))


def test_pinning_a_bone_offset_that_contradicts_the_measured_map_warns():
    """`p3rd_anim_map.BONE_OFFSET[40] == 0`. Forking that in a manifest is exactly how
    a port builds at one bone map while the renderer draws it at another."""
    body = BASE + "\nbone_offset = 2\n"
    iss = V.validate(MF.loads(body), pac=None)
    assert "BONE_OFFSET_OVERRIDE" in _codes(iss), iss
    assert "BONE_OFFSET_OVERRIDE" not in _codes(
        V.validate(MF.loads(BASE + "\nbone_offset = 0\n"), pac=None))


def test_the_shipped_ports_have_no_errors_without_any_evidence():
    for m in MF.discover(os.path.join(_ROOT, "ports")):
        iss = V.validate(m, pac=None)
        assert not [i for i in iss if i.level == V.ERROR], (m.name, iss)


def test_a_clip_that_moved_slots_is_told_where_it_went():
    """🔴 The #8 addition: "wrong clip in slot 60" is not actionable; "your 382f clip
    is now at slot 61" is. Clip ids shift on every rebuild."""
    m = MF.loads(BASE + '''
[clips.charge]
slot = 60
frames = 382
loop = false
''')
    real = V.clip_table
    V.clip_table = lambda blob: {1: (180, True), 60: (252, False), 61: (382, False)}
    try:
        codes = {i.code: i for i in V.validate(m, pac=b"x")}
    finally:
        V.clip_table = real
    got = codes["CLIP_FRAMES_MISMATCH"]
    assert "now at slot 61, not 60" in got.message, got.message



# --------------------------------------------------------------------------- #
# the part system (#10)
# --------------------------------------------------------------------------- #
def _grid(state="normal", cell=None):
    rows = [[0] * 10 for _ in range(7)]
    if cell:
        r, c, v = cell
        rows[r][c] = v
    return ('\n[[hitzone]]\nstate = "%s"\nrows = [\n%s\n]\n'
            % (state, "\n".join("  [%s]," % ", ".join(map(str, r)) for r in rows)))


def _report(m, **kw):
    """Codes from a full validate() run — the part checks need the manifest, not a
    pre-built issue list, so this is a different helper from `_codes` above."""
    return {i.code for i in V.validate(m, **kw)}


def test_two_names_for_one_part_index_is_an_error():
    """The index IS the engine's break slot. Two names for slot 1 means two break
    bars in the UI that are secretly one bar in the game."""
    m = MF.loads(BASE + "\n[parts.head]\nindex = 1\n[parts.face]\nindex = 1\n")
    assert "PART_INDEX_DUPLICATE" in _report(m)


def test_a_hurtbox_with_no_part_warns_rather_than_silently_becoming_part_zero():
    m = MF.loads(BASE + "\n[[hurtbox]]\nbone = 10\nradius = 150.0\n")
    assert "HURTBOX_NO_PART" in _report(m)


def test_a_capsule_without_a_far_end_is_an_error():
    m = MF.loads(BASE + '\n[[hurtbox]]\nbone=1\nradius=1.0\nshape="capsule"\n')
    assert "HURTBOX_CAPSULE_NO_END" in _report(m)


def test_an_all_zero_grid_warns_that_nothing_can_hurt_him():
    m = MF.loads(BASE + _grid())
    assert "HITZONE_ALL_ZERO" in _report(m)


def test_two_states_with_the_same_name_is_an_error():
    """The engine picks a state by INDEX; the name is the only thing a human has."""
    m = MF.loads(BASE + _grid("rage", (0, 1, 50)) + _grid("rage", (0, 1, 60)))
    assert "HITZONE_STATE_DUPLICATE" in _report(m)


def test_authoring_a_grid_always_says_it_is_shared_and_unproven():
    """🔴 Two facts the author cannot be allowed to forget: the grid is species data
    so it changes the native host too, and no cold boot has ever confirmed that
    writing it does anything (#19)."""
    m = MF.loads(BASE + _grid("normal", (0, 1, 75)))
    issues = [i for i in V.validate(m) if i.code == "HITZONE_SHARED_AND_UNVALIDATED"]
    assert len(issues) == 1
    assert "native host" in issues[0].message and "#19" in issues[0].message


def test_more_states_than_the_host_ships_warns_against_the_intel():
    si = INTEL.find_intel(75)
    if si is None or not si.parts.has_grid:
        return
    m = MF.loads(BASE + _grid("a", (0, 1, 5)) + _grid("b", (0, 1, 6))
                 + _grid("c", (0, 1, 7)))
    assert "HITZONE_STATE_COUNT" in _report(m, intel=si)
    two = MF.loads(BASE + _grid("a", (0, 1, 5)) + _grid("b", (0, 1, 6)))
    assert "HITZONE_STATE_COUNT" not in _report(two, intel=si)


def test_hurtboxes_with_no_named_parts_warns_once():
    m = MF.loads(BASE + "\n[[hurtbox]]\nbone=1\nradius=1.0\npart=1\n")
    assert "PARTS_UNNAMED" in _report(m)
    named = MF.loads(BASE + "\n[parts.head]\nindex=1\n"
                     "\n[[hurtbox]]\nbone=1\nradius=1.0\npart=1\n")
    codes = _report(named)
    assert "PARTS_UNNAMED" not in codes and "HURTBOX_PART_UNNAMED" not in codes


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
