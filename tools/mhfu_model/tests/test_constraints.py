"""Phase 2 guard: the constraint validator.

Acceptance: a clean monster passes; deliberately breaking each rule produces the
matching ERROR with a specific, correct code.

Run: PYTHONPATH=tools python tools/mhfu_model/tests/test_constraints.py
"""
from __future__ import annotations

import copy
import os

from mhfu_model import load_pac
from mhfu_model import constraints as C
from mhfu_model.model import Bone, Keyframe

DATA = os.path.join(os.path.dirname(__file__), "..", "..", "..",
                    "workspace", "extracted", "data_files")
TIGREX = os.path.join(DATA, "file_06134.bin")


def _codes(rep):
    return {r.code for r in rep.results}


def test_clean_fixture_passes():
    mm = load_pac(TIGREX)
    C.register_template("tigrex", mm.skeleton)
    rep = C.validate(mm.model, mm.skeleton, mm.anim, target_species="tigrex")
    assert rep.ok, "clean Tigrex flagged errors: %s" % [str(r) for r in rep.errors]
    # only benign advisories allowed (e.g. mesh-group count vs bones)
    assert C.ERROR not in {r.level for r in rep.results} or not rep.errors


def test_added_bone_breaks_template():
    """Adding a skeleton bone past the species budget is flagged against the
    template (an extra unanimated joint is not itself an engine error)."""
    mm = load_pac(TIGREX)
    C.register_template("tigrex", mm.skeleton)
    sk = copy.deepcopy(mm.skeleton)
    sk.bones.append(Bone(index=len(sk.bones), parent=0, child=-1, sibling=-1,
                         bind_scale=(1, 1, 1), bind_rot=(0, 0, 0),
                         bind_pos=(0, 0, 0), raw=b"\x00" * 0x10C))
    sk.bone_count += 1
    rep = C.validate(mm.model, sk, mm.anim, target_species="tigrex")
    assert "TEMPLATE_BONE_COUNT" in _codes(rep)
    assert not rep.ok


def test_removed_bones_break_lockstep():
    """Cutting the skeleton below the animated bone count is a hard error."""
    mm = load_pac(TIGREX)
    n_tracks = len(mm.anim.animations[0].tracks)
    sk = copy.deepcopy(mm.skeleton)
    sk.bones = sk.bones[: n_tracks - 3]
    sk.bone_count = len(sk.bones)
    rep = C.validate(mm.model, sk, mm.anim)
    assert "BONE_COUNT_MISMATCH" in _codes(rep)
    assert not rep.ok


def test_anim_track_desync():
    """One anim animating a different bone set than the rest is a desync error."""
    mm = load_pac(TIGREX)
    mm.anim.animations[0].tracks = mm.anim.animations[0].tracks[:-2]
    mm.anim.animations[0].bone_count = len(mm.anim.animations[0].tracks)
    rep = C.validate(mm.model, mm.skeleton, mm.anim)
    assert "ANIM_TRACK_DESYNC" in _codes(rep)
    assert not rep.ok


def test_out_of_range_rotation_flagged():
    mm = load_pac(TIGREX)
    a = next(a for a in mm.anim.animations
             for tr in a.tracks for ch in tr.channels if ch.keyframes)
    tr = next(t for t in a.tracks for ch in t.channels if ch.keyframes)
    ch = next(c for c in tr.channels if c.keyframes)
    ch.keyframes[0] = Keyframe(value=70000, frame=0, ease_in=0, ease_out=0)
    rep = C.validate(mm.model, mm.skeleton, mm.anim)
    assert "S16_OVERFLOW" in _codes(rep)
    assert not rep.ok


def test_unbound_mesh_group_flagged():
    mm = load_pac(TIGREX)
    # break draw-order contiguity (simulate an unbound / misordered group)
    mm.model.mesh_groups[3].index = 99
    rep = C.validate(mm.model, mm.skeleton, mm.anim)
    assert "MESH_ORDER" in _codes(rep)
    assert not rep.ok


def test_channel_mask_mismatch_flagged():
    mm = load_pac(TIGREX)
    tr = next(t for a in mm.anim.animations for t in a.tracks if t.channels)
    tr.channels.append(copy.deepcopy(tr.channels[0]))   # more channels than mask bits
    rep = C.validate(mm.model, mm.skeleton, mm.anim)
    assert "CHANNEL_MASK_COUNT" in _codes(rep)
    assert not rep.ok


def test_unknown_species_warns_not_errors():
    mm = load_pac(TIGREX)
    rep = C.validate(mm.model, mm.skeleton, mm.anim, target_species="nonesuch")
    assert "NO_TEMPLATE" in _codes(rep)
    assert rep.ok, "an unknown template is a warning, not an error"


def test_advisories_warn():
    mm = load_pac(TIGREX)
    rep = C.validate(mm.model, mm.skeleton, mm.anim,
                     advisories={"new_anim_slots": [50], "extra_combatants": 3})
    assert "NEW_CLIP_NEEDS_AI" in _codes(rep)
    assert "COMBAT_CAP" in _codes(rep)
    assert rep.ok, "advisories are warnings, not errors"


if __name__ == "__main__":
    test_clean_fixture_passes()
    test_added_bone_breaks_template()
    test_removed_bones_break_lockstep()
    test_anim_track_desync()
    test_out_of_range_rotation_flagged()
    test_unbound_mesh_group_flagged()
    test_channel_mask_mismatch_flagged()
    test_unknown_species_warns_not_errors()
    test_advisories_warn()
    print("OK — validator: clean passes; each broken rule flagged with the right code")
