"""The golden test: the vectorised FK must reproduce `mhfu_model.stretch`, term for term.

🔴 **`tools/mhfu_model/stretch.py` is the ORACLE.** Its scalar FK and blend skinning are
what the project validated to 0.00 units on all 46 driven joints of every clip against a
built PAC; `mhfu_monster_editor.core.pose` is the same maths with the Python loops
replaced by array ops. A numpy rewrite whose only oracle is itself is worthless, so every
number below is a comparison against the scalar path on the SAME data — the native
Tigrex, the built Zinogre port and the MHP3rd donor.

Measured on this data (the assertions leave four orders of headroom):

    world matrices, every clip x 5 frames   max |Δ|  4.6e-13   (530 poses, both games)
    skinned vertices, every mesh group      max |Δ|  2.6e-13   (4128 vertices)
    dominant joint per vertex               identical, 4128/4128
    bind joints vs convert.bone_world_positions       exactly 0.0

⚠️ Needs the game extracts under `workspace/`. The tests that do return early without
them, like `tools/mhfu_model/tests/test_stream_partition.py`; the synthetic ones below
(channel clamping, a cyclic parent) run everywhere and are the reason this file is not
vacuous on a fresh checkout.
"""
import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, _ROOT)
sys.path.insert(0, os.path.join(_ROOT, "tools"))

import numpy as np

from mhfu_monster_editor.core import pose as P
from mhfu_monster_editor.core import open_scene

import mhfu_model
from mhfu_model import convert as C
from mhfu_model import stretch as ST
from mhfu_model.model import (Animation, Bone, BoneTrack, Channel, Keyframe,
                              Skeleton)

MHFU = os.path.join(_ROOT, "workspace", "extracted", "data_files")
MHP3 = os.path.join(_ROOT, "workspace", "extracted_mhp3", "data_files")
TIGREX = os.path.join(MHFU, "file_06185.bin")
ZINOGRE_SRC = os.path.join(MHP3, "file_05339.bin")

#: four orders of headroom over the worst deviation measured (4.6e-13 on matrices).
#: World coordinates here run to ~1500 units, so this is ~1e-12 relative.
TOL = 1e-9


def _have(path):
    return os.path.exists(path)


# --------------------------------------------------------------------------- #
# synthetic — these run without any game data
# --------------------------------------------------------------------------- #
def _skeleton(parents, pos=None):
    bones = [Bone(index=i, parent=p, child=-1, sibling=-1, bind_scale=(1.0, 1.0, 1.0),
                  bind_rot=(0.0, 0.0, 0.0),
                  bind_pos=pos[i] if pos else (10.0 * (i + 1), 2.0, -3.0))
             for i, p in enumerate(parents)]
    return Skeleton(bone_count=len(bones), total_size=0, bones=bones)


def _animation(per_track):
    """`per_track` = [{channel_bit: [(frame, raw_s16), ...]}, ...] -> an Animation."""
    tracks = []
    for chans in per_track:
        tracks.append(BoneTrack(tag=0x80000000, channels=[
            Channel(type=0x80000000 | bit,
                    keyframes=[Keyframe(value=v, frame=f, ease_in=0, ease_out=0)
                               for f, v in keys])
            for bit, keys in sorted(chans.items())]))
    return Animation(slot=0, tag=0, bone_count=len(tracks), loop=0, loop_start=0.0,
                     tracks=tracks)


def test_a_channel_clamps_at_its_own_last_key_not_the_clips():
    """The trap this pins: a joint that stops keying at frame 10 must HOLD that value
    while its neighbours keep moving to frame 30. Clamping per clip instead — or
    letting a channel run off the end of its key list — bends one limb and not the
    others, and looks exactly like a bad rig."""
    skel = _skeleton([-1, 0, 1])
    anim = _animation([
        {},                                            # joint 0: nothing
        {0x008: [(0, 0), (10, 4096)]},                 # joint 1: rotX, stops at 10
        {0x008: [(0, 0), (30, 2048)]},                 # joint 2: rotX, runs to 30
    ])
    samples = P.sample(anim)
    assert samples == ST.sample(anim), "the sampler drifted from stretch.sample"
    rig = P.Rig.from_skeleton(skel)
    curves = P.Curves(samples, rig.n_bones, rig.bind_local)
    for frame in (0, 5, 10, 20, 30, 99):
        rot, loc = curves.eval(frame)
        got = rig.world(rot, loc)
        ref = ST.world_matrices(skel, samples, frame)
        err = np.abs(got - np.array([ref[i] for i in range(rig.n_bones)])).max()
        assert err < TOL, "frame %s: %g" % (frame, err)
    # and the clamp really is per channel: at frame 25 joint 1 HOLDS the 90 deg it
    # reached at frame 10, while joint 2 is still 5/6 of the way to its own 45 deg.
    rot = curves.eval(25)[0]
    assert abs(rot[1, 0] - np.pi / 2) < 1e-12, rot[1, 0]
    assert abs(rot[2, 0] - (25.0 / 30.0) * np.pi / 4) < 1e-12, rot[2, 0]


def test_an_undriven_channel_falls_back_to_the_bind_offset():
    """`loc` defaults to the joint's own bind_pos per AXIS, `rot` to zero. Defaulting
    loc to zero instead collapses every undriven joint onto its parent."""
    skel = _skeleton([-1, 0], pos=[(1.0, 2.0, 3.0), (4.0, 5.0, 6.0)])
    anim = _animation([{}, {0x080: [(0, 160), (10, 320)]}])     # locY only, on joint 1
    rig = P.Rig.from_skeleton(skel)
    curves = P.Curves(P.sample(anim), rig.n_bones, rig.bind_local)
    _rot, loc = curves.eval(0)
    assert tuple(loc[1]) == (4.0, 10.0, 6.0), loc[1]     # X and Z keep bind, Y is keyed
    assert tuple(loc[0]) == (1.0, 2.0, 3.0), loc[0]


def test_a_cyclic_parent_becomes_a_root_exactly_as_the_reference_does():
    """`stretch.world_matrices` breaks a parent cycle by treating the joint it re-enters
    as a root. The vectorised FK cannot recurse, so it resolves the same forest ONCE up
    front — this pins the two answers together on the case that would otherwise hang."""
    skel = _skeleton([-1, 2, 1])                # 1 -> 2 -> 1 is a cycle
    parents = P.effective_parents(skel)
    assert list(parents) == [-1, 2, -1], list(parents)
    rig = P.Rig.from_skeleton(skel)
    ref = ST.world_matrices(skel, None, 0)
    err = np.abs(rig.bind_world - np.array([ref[i] for i in range(3)])).max()
    assert err < TOL, err


def test_an_unskinned_vertex_stays_at_bind():
    """A PMO palette pads unused slots with weight 0.0, and a group can have no palette
    at all. `stretch.skin_group` leaves such a vertex where it is; so must this, or the
    whole group collapses onto joint 0."""
    b = P.SkinBinding.from_vertices(
        [{"x": 1.0, "y": 2.0, "z": 3.0, "influences": [(0, 0.0), (1, 0.0)]},
         {"x": 4.0, "y": 5.0, "z": 6.0}], n_bones=2)
    assert list(b.dominant()) == [-1, -1], b.dominant()
    deform = np.tile(np.eye(4), (2, 1, 1))
    deform[:, :3, 3] = 100.0                    # any deform at all
    assert np.abs(b.apply(deform) - b.positions).max() == 0.0


# --------------------------------------------------------------------------- #
# against the scalar reference, on the real files
# --------------------------------------------------------------------------- #
def _fk_sweep(scene, label, frames_per_clip=5):
    """Every clip, several frames: numpy world matrices vs `stretch.world_matrices`."""
    worst, checked = 0.0, 0
    for clip in scene.clips:
        samples = P.sample(clip._anim, clip.track_to_joint or None)
        last = max(clip.frames, 1)
        for frac in np.linspace(0.0, 1.2, frames_per_clip):
            frame = int(last * frac)
            got = scene.pose(clip, frame).world
            ref = ST.world_matrices(scene.skeleton, samples, frame)
            err = np.abs(got - np.array([ref[i] for i in range(scene.rig.n_bones)])).max()
            worst = max(worst, float(err))
            checked += 1
    print("      %-28s %4d poses, worst |Δ| %.2e" % (label, checked, worst))
    assert worst < TOL, "%s: %g" % (label, worst)
    return worst


def test_the_fk_reproduces_the_scalar_reference_on_the_native_tigrex():
    if not _have(TIGREX):
        return                      # extracts are gitignored; docs/ASSETS.md rebuilds
    _fk_sweep(open_scene(TIGREX), "native Tigrex, 64 clips")


def test_the_fk_reproduces_the_scalar_reference_on_the_mhp3rd_donor():
    """🔴 Fed with the SAME record->bone map on both sides, so this pins the FK and not
    the mapping. The mapping is pinned separately, against the built port."""
    if not _have(ZINOGRE_SRC):
        return
    _fk_sweep(open_scene(ZINOGRE_SRC), "MHP3rd Zinogre, 42 clips")


def test_the_sampler_is_the_scalar_sampler_for_a_whole_rig_mhfu_clip():
    """`stretch.sample` is positional and that is RIGHT for MHFU's own clips and for a
    built port. Where this loader departs from it — a partial clip, an MHP3rd donor —
    it is because positional is wrong there, never as an accident."""
    if not _have(TIGREX):
        return
    sc = open_scene(TIGREX)
    whole = [c for c in sc.clips if c.whole_rig]
    assert len(whole) == 62, len(whole)
    for clip in whole:
        assert clip.track_to_joint == {i: i for i in range(clip.tracks)}, clip.slot
        assert P.sample(clip._anim, clip.track_to_joint) == ST.sample(clip._anim)


def test_the_skinning_reproduces_the_scalar_reference_vertex_for_vertex():
    if not _have(TIGREX):
        return
    sc = open_scene(TIGREX)
    raw = mhfu_model.load_pac(TIGREX).model.mesh_groups
    assert len(raw) == len(sc.groups)
    clip = sc.clip(7)
    frame = clip.frames // 2
    samples = P.sample(clip._anim, clip.track_to_joint or None)
    deform_ref = ST.deform_matrices(sc.skeleton, samples, frame)
    deform = sc.pose(clip, frame).deform
    worst, verts, mismatched = 0.0, 0, 0
    for g, group in zip(raw, sc.groups):
        ref = ST.skin_group(g, deform_ref)
        if not ref:
            continue
        got = group.skin.apply(deform)
        want = np.array([r[0] for r in ref], dtype=np.float64)
        worst = max(worst, float(np.abs(got - want).max()))
        mismatched += int((group.skin.dominant()
                           != np.array([r[1] for r in ref], dtype=np.int64)).sum())
        verts += len(ref)
    print("      %-28s %4d vertices, worst |Δ| %.2e, %d dominant-joint mismatches"
          % ("native Tigrex skinning", verts, worst, mismatched))
    assert verts == 4128, verts
    assert worst < TOL, worst
    assert mismatched == 0, mismatched


def test_the_bind_pose_matches_an_independent_accumulation():
    """`convert.bone_world_positions` walks the same tree with different code and no
    matrices at all. Agreeing with it EXACTLY is the cheapest guard there is against a
    parent array that is subtly wrong."""
    for path in (TIGREX, ZINOGRE_SRC):
        if not _have(path):
            continue
        sc = open_scene(path)
        ref = C.bone_world_positions(sc.skeleton)
        want = np.array([ref[i] for i in range(sc.rig.n_bones)], dtype=np.float64)
        err = float(np.abs(sc.rig.bind_joints - want).max())
        assert err == 0.0, "%s: %g" % (os.path.basename(path), err)


def test_posing_at_bind_leaves_the_geometry_where_it_was():
    """The deform of the bind pose is the identity, so every vertex must land on itself
    — including the ones blended across four joints."""
    if not _have(TIGREX):
        return
    sc = open_scene(TIGREX)
    moved = sc.bind_pose().skin(sc.merged)
    err = float(np.abs(moved - sc.merged.positions).max())
    assert err < 1e-9, err


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    if not _have(TIGREX):
        print("  workspace/ extracts absent — the reference sweeps are skipped "
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
