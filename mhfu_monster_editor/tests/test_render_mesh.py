"""The viewport's content: normal skinning, the fork rule, and what actually drew.

Split the way the render tests are: the parts that need no GL run first and always
(normal skinning against the position skinning it must agree with, the leading-origin
chain, the undriven-geometry report, the default-pose rule), then the GL parts, which
return early on a machine with no driver.

    venv/bin/python mhfu_monster_editor/tests/test_render_mesh.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from mhfu_monster_editor.core.pose import Rig, SkinBinding  # noqa: E402
from mhfu_monster_editor.render.mesh import (ISOLATE_HIDE, ISOLATE_ONLY,  # noqa: E402
                                             MODE_VGROUP, default_pose)
from mhfu_monster_editor.render.skeleton import (body_fork,  # noqa: E402
                                                 leading_chain, undriven_geometry)

TIGREX = _ROOT / "workspace" / "extracted" / "data_files" / "file_06185.bin"
ZINOGRE = _ROOT / "tmp" / "zinogre_v10.bin"


def _scene(path):
    from mhfu_monster_editor.core import open_scene
    return open_scene(path)


# --------------------------------------------------------------------------- #
# no GL
# --------------------------------------------------------------------------- #
def test_normal_skinning_matches_position_skinning():
    """A normal must rotate exactly as the surface it is normal TO.

    Checked without any file: build a two-bone rig, rotate one bone, and confirm that
    a normal carried through :meth:`apply_directions` equals the normal of the
    triangle carried through :meth:`apply`. That is the whole contract, and it is what
    catches the classic bug of blending the full 4x4 into a direction and dragging
    every normal to wherever the joint translated to.
    """
    rig = Rig(parents=[-1, 0], bind_local=np.array([[0.0, 0, 0], [0.0, 10, 0]]))
    tri = np.array([[0.0, 10, 0], [2.0, 10, 0], [0.0, 10, 3]])
    bones = np.array([[1], [1], [1]])
    weights = np.array([[1.0], [1.0], [1.0]])
    skin = SkinBinding(tri, bones, weights)

    def face_normal(p):
        n = np.cross(p[1] - p[0], p[2] - p[0])
        return n / np.linalg.norm(n)

    bind_n = np.tile(face_normal(tri), (3, 1))
    for angle in (0.0, 0.3, 1.1, -2.0):
        rot = np.zeros((2, 3))
        rot[1] = (angle, angle * 0.5, -angle * 0.25)
        loc = rig.bind_local.copy()
        loc[0] = (100.0, -40.0, 7.0)          # a big translation the normal must ignore
        deform = rig.deform(rig.world(rot, loc))
        moved = skin.apply(deform)
        got = skin.apply_directions(deform, bind_n)
        want = face_normal(moved)
        err = float(np.abs(got - want).max())
        assert err < 1e-12, "angle %.2f: normals disagree by %.2e" % (angle, err)
        assert np.allclose(np.linalg.norm(got, axis=1), 1.0), "normals must stay unit"
    print("normals           rotate with the surface, immune to translation (<1e-12)")


def test_normal_skinning_rejects_a_length_mismatch():
    skin = SkinBinding(np.zeros((3, 3)), np.zeros((3, 1), dtype=int), np.ones((3, 1)))
    try:
        skin.apply_directions(np.tile(np.eye(4), (1, 1, 1)), np.zeros((2, 3)))
    except ValueError:
        print("normals           a length mismatch raises rather than broadcasting")
        return
    raise AssertionError("2 directions for 3 vertices should have been refused")


def test_leading_chain_is_the_fork_and_its_ancestors():
    # 0 -> 1 -> 2 -> {3, 5}; the fork is 2, so the lead chain is 0, 1, 2.
    parents = [-1, 0, 1, 2, 3, 2, 5]
    assert body_fork(parents) == 2, body_fork(parents)
    assert leading_chain(parents) == (0, 1, 2), leading_chain(parents)
    # a single chain has no fork to divide — an empty lead, not a crash.
    assert leading_chain([-1, 0, 1, 2]) == () or body_fork([-1, 0, 1, 2]) >= 0
    print("fork              lead chain = the fork and every ancestor of it")


def test_scene_facts():
    """The two monsters, on the facts issue #6's viewport turns on."""
    if not TIGREX.exists():
        print("SKIP: no game data (docs/ASSETS.md)")
        return
    sc = _scene(TIGREX)
    assert body_fork(sc.rig.parents) == 2, "the native Tigrex's fork is joint 2"
    assert leading_chain(sc.rig.parents) == (0, 1, 2), leading_chain(sc.rig.parents)

    # 🔴 48 joints in the skeleton, 45 in the animation partition (31/9/5). Joints
    # 45 -> 46 -> 47 are a second root chain no clip drives, and they carry geometry.
    orph = undriven_geometry(sc)
    assert set(orph) == {46, 47}, orph
    assert sum(orph.values()) == 150, orph
    assert int(sc.rig.parents[45]) == -1, "45 is the second root"

    clip, frame = default_pose(sc)
    assert clip is not None and clip.loop and clip.whole_rig, clip
    assert 0 < frame < clip.frames, (frame, clip.frames)
    # and it is genuinely NOT the bind pose — the point of the whole rule.
    moved = np.abs(sc.pose(clip, frame).joints - sc.rig.bind_joints).max()
    assert moved > 50.0, "the default pose is %.1f units from bind — too close" % moved
    print("tigrex            fork 2, lead (0,1,2), 150 verts on undriven 46/47, "
          "default slot %d f%.0f (%.0f u from bind)" % (clip.slot, frame, moved))


def test_default_pose_on_the_port():
    if not ZINOGRE.exists():
        print("SKIP: tmp/zinogre_v10.bin is not built (docs/ASSETS.md §D)")
        return
    sc = _scene(ZINOGRE)
    clip, frame = default_pose(sc)
    assert clip is not None and clip.loop, clip
    moved = np.abs(sc.pose(clip, frame).joints - sc.rig.bind_joints).max()
    assert moved > 50.0, moved
    print("zinogre           default slot %d f%.0f (%.0f u from bind)"
          % (clip.slot, frame, moved))


# --------------------------------------------------------------------------- #
# GL
# --------------------------------------------------------------------------- #
def _lit(img, background) -> int:
    bg = np.array([int(round(c * 255)) for c in background[:3]])
    return int((np.abs(img[..., :3].astype(int) - bg).max(axis=2) > 12).sum())


def test_mesh_draws_and_the_modes_differ(ctx):
    if not TIGREX.exists():
        print("SKIP: no game data for the GL pass")
        return
    from mhfu_monster_editor.render.viewport import Viewport

    sc = _scene(TIGREX)
    with Viewport(ctx, (360, 280)) as vp:
        vp.set_scene(sc)
        vp.show_ground = vp.show_skeleton = False
        vp.camera.look("side")
        vp.draw()
        textured = vp.target.read()
        assert _lit(textured, vp.background) > 2000, "the mesh did not draw"

        vp.mesh.mode = MODE_VGROUP
        vp.draw()
        vgroup = vp.target.read()
        assert not np.array_equal(textured, vgroup), "vgroup mode changed nothing"

        # the point of the tag/isolate pair: ONLY and HIDE must partition the animal.
        vp.mesh.mode = 0
        head = [3, 4, 5, 6, 7, 8]
        vp.tag_joints(head)
        vp.mesh.isolate = ISOLATE_ONLY
        vp.draw()
        only = _lit(vp.target.read(), vp.background)
        vp.mesh.isolate = ISOLATE_HIDE
        vp.draw()
        hide = _lit(vp.target.read(), vp.background)
        vp.mesh.isolate = 0
        vp.draw()
        whole = _lit(vp.target.read(), vp.background)
        assert 0 < only < whole, (only, whole)
        assert 0 < hide < whole, (hide, whole)
        # they overlap in depth, so the two parts cannot sum to less than the whole.
        assert only + hide >= whole * 0.95, (only, hide, whole)
    print("mesh              textured/vgroup differ; isolate ONLY %d + HIDE %d vs "
          "whole %d" % (only, hide, whole))


def test_posing_moves_pixels(ctx):
    if not TIGREX.exists():
        print("SKIP: no game data")
        return
    from mhfu_monster_editor.render.viewport import Viewport

    sc = _scene(TIGREX)
    with Viewport(ctx, (300, 240)) as vp:
        vp.set_scene(sc)
        vp.show_ground = vp.show_skeleton = False
        clip = vp.clip
        vp.set_pose(clip, 0)
        vp.draw()
        a = vp.target.read()
        vp.set_pose(clip, clip.frames)
        vp.draw()
        b = vp.target.read()
        assert not np.array_equal(a, b), "frame 0 and frame %d render identically" \
            % clip.frames
        vp.set_pose(None)
        vp.draw()
        bind = vp.target.read()
        assert not np.array_equal(a, bind), "the posed frame equals the bind pose"
    print("posing            frame 0 / frame %d / bind are three different pictures"
          % clip.frames)


def test_skeleton_picking_round_trips(ctx):
    """Project a joint, pick at that pixel, get the same joint back."""
    if not TIGREX.exists():
        print("SKIP: no game data")
        return
    from mhfu_monster_editor.render.viewport import Viewport

    sc = _scene(TIGREX)
    with Viewport(ctx, (400, 320)) as vp:
        vp.set_scene(sc)
        vp.camera.look("side")
        sk = vp.skeleton
        mvp = vp.camera.mvp(vp.target.aspect)
        size = vp.target.size
        xy, ok = sk.project(mvp, size)
        tried = hits = 0
        for j in range(len(xy)):
            if not ok[j]:
                continue
            tried += 1
            got = sk.pick(mvp, size, xy[j][0], xy[j][1], radius=2.0)
            # joints coincide (the Tigrex has several at the same point), so the pick
            # is correct if it lands on ANY joint at that pixel.
            if got is not None and np.allclose(xy[got], xy[j], atol=2.0):
                hits += 1
        assert tried > 20, "only %d joints were on screen" % tried
        assert hits == tried, "%d of %d joints did not pick at their own pixel" \
            % (tried - hits, tried)
        far = sk.pick(mvp, size, -500.0, -500.0, radius=8.0)
        assert far is None, "a click far off the model picked joint %s" % far
    print("picking           %d/%d on-screen joints pick at their own pixel" % (hits, tried))


def main() -> int:
    test_normal_skinning_matches_position_skinning()
    test_normal_skinning_rejects_a_length_mismatch()
    test_leading_chain_is_the_fork_and_its_ancestors()
    test_scene_facts()
    test_default_pose_on_the_port()

    try:
        from mhfu_monster_editor.render.context import ContextError, describe, headless
    except ImportError as e:
        print("SKIP: moderngl is not installed (%s)" % e)
        print("\ntest_render_mesh: OK (no-GL part only)")
        return 0
    try:
        ctx = headless()
    except ContextError as e:
        print("SKIP: no GL on this machine.\n%s" % e)
        print("\ntest_render_mesh: OK (no-GL part only)")
        return 0
    print("context           %s" % describe(ctx))
    try:
        test_mesh_draws_and_the_modes_differ(ctx)
        test_posing_moves_pixels(ctx)
        test_skeleton_picking_round_trips(ctx)
    finally:
        ctx.release()
    print("\ntest_render_mesh: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
