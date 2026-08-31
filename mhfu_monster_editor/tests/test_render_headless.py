"""The headless path: a standalone context, a real frame, a real PNG.

Needs a GL driver and so **returns early without one** — the same contract
`test_ports_build.py` and `test_core_scene.py` use for missing game data. A machine
that cannot render says so and exits 0; it does not fail the suite. The maths that CAN
be checked everywhere is in `test_render_camera.py`, which needs no driver at all.

    venv/bin/python mhfu_monster_editor/tests/test_render_headless.py
"""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from mhfu_monster_editor.render.camera import Bounds  # noqa: E402
from mhfu_monster_editor.render.context import ContextError  # noqa: E402
from mhfu_monster_editor.render.target import write_png, _write_png_stdlib  # noqa: E402

#: a native Tigrex, if this machine has the extracts. `docs/ASSETS.md` — never committed.
TIGREX = _ROOT / "workspace" / "extracted" / "data_files" / "file_06185.bin"


def _context():
    """A standalone context, or None with a printed reason."""
    try:
        from mhfu_monster_editor.render.context import describe, headless
    except ImportError as e:
        print("SKIP: moderngl is not installed (%s)" % e)
        return None
    try:
        ctx = headless()
    except ContextError as e:
        print("SKIP: no GL on this machine.\n%s" % e)
        return None
    print("context           %s" % describe(ctx))
    return ctx


# --------------------------------------------------------------------------- #
# no GL needed
# --------------------------------------------------------------------------- #
def test_png_encoders_agree():
    """The stdlib fallback must produce the same image Pillow does.

    It exists so ``--headless out.png`` cannot fail for want of an *image* library
    when the *graphics* stack is fine — which is only worth anything if the bytes it
    writes decode to the same pixels.
    """
    rng = np.random.default_rng(3)
    img = rng.integers(0, 256, (23, 41, 4), dtype=np.uint8)
    with tempfile.TemporaryDirectory() as d:
        a, b = Path(d) / "pil.png", Path(d) / "std.png"
        write_png(a, img)
        _write_png_stdlib(b, img)
        try:
            from PIL import Image
        except ImportError:
            print("png               stdlib encoder wrote %d bytes (no Pillow to "
                  "compare against)" % b.stat().st_size)
            return
        for p in (a, b):
            got = np.asarray(Image.open(p).convert("RGBA"))
            assert got.shape == img.shape, (p.name, got.shape)
            assert np.array_equal(got, img), "%s does not round-trip" % p.name
    print("png               Pillow and the stdlib fallback round-trip identically")


# --------------------------------------------------------------------------- #
# GL
# --------------------------------------------------------------------------- #
def test_hurtbox_surfaces_are_exact():
    """The filled shell's maths, with no GL at all.

    A capsule here is ONE sphere lattice whose centre moves with latitude — `b` above
    the equator, `a` below, with the equator ring duplicated so the quads between the
    two copies form the tube. That is compact and easy to get subtly wrong, so the
    check is the definition: every vertex is exactly `radius` from the segment a..b.
    """
    from mhfu_monster_editor.render.hitboxes import (capsule_surface,
                                                     sphere_surface)

    s = sphere_surface((10.0, -5.0, 2.0), 100.0)
    assert len(s) % 3 == 0 and len(s) > 100
    r = np.linalg.norm(s - np.array([10.0, -5.0, 2.0]), axis=1)
    assert np.allclose(r, 100.0), (r.min(), r.max())

    a, b = np.array([0.0, 0.0, 0.0]), np.array([30.0, 0.0, 300.0])
    c = capsule_surface(a, b, 50.0)
    ab = b - a
    t = np.clip(((c - a) @ ab) / float(ab @ ab), 0.0, 1.0)[:, None]
    d = np.linalg.norm(c - (a + t * ab), axis=1)
    assert np.allclose(d, 50.0), (d.min(), d.max())
    # and it really is a capsule, not two loose spheres. The tube carries no vertices
    # of its own — it is the quads BETWEEN the two duplicated equator rings — so the
    # check is that some triangle spans the whole segment.
    tri = t.reshape(-1, 3)
    spans = (tri.min(axis=1) < 0.05) & (tri.max(axis=1) > 0.95)
    assert spans.any(), "the two hemispheres are not joined by a tube"

    # a zero-length capsule is a sphere, not a division by zero
    assert len(capsule_surface((1, 2, 3), (1, 2, 3), 10.0)) == len(sphere_surface(
        (1, 2, 3), 10.0))
    print("hurtbox maths     sphere + capsule surfaces exact to the radius (no GL)")


def test_target_reads_back_what_was_cleared(ctx):
    from mhfu_monster_editor.render.target import Target

    with Target(ctx, (64, 48), samples=4) as t:
        t.use()
        t.clear((1.0, 0.5, 0.25, 1.0))
        img = t.read()
        assert img.shape == (48, 64, 4), img.shape
        assert abs(int(img[0, 0, 0]) - 255) <= 1 and abs(int(img[0, 0, 1]) - 128) <= 2, \
            img[0, 0]
        assert t.resize((100, 70)) and t.size == (100, 70)
        assert not t.resize((100, 70)), "resize to the same size must be a no-op"
        assert t.resize((0, 0)) and t.size == (1, 1), "a collapsed panel must clamp"
    print("target            clear/read/resize/msaa-resolve ok (%dx msaa)"
          % Target(ctx, (8, 8)).samples)


def test_read_is_top_down(ctx):
    """🔴 GL's row 0 is the bottom; every other image in this repo puts it at the top.

    Painted by clearing the whole buffer and then scissoring the TOP half to a second
    colour, so the assertion is about orientation and nothing else.
    """
    from mhfu_monster_editor.render.target import Target

    with Target(ctx, (16, 16), samples=0) as t:
        t.use()
        t.clear((0.0, 0.0, 0.0, 1.0))
        ctx.scissor = (0, 8, 16, 8)          # GL y=8..16 == the TOP half on screen
        t.clear((1.0, 1.0, 1.0, 1.0))
        ctx.scissor = None
        img = t.read()
        assert img[0, 0, 0] > 200, "row 0 should be the white TOP half, got %s" % img[0, 0]
        assert img[-1, 0, 0] < 55, "the last row should be the black bottom"
    print("target            read() flips GL's bottom-up buffer to top-down")


def test_viewport_draws_the_scene(ctx):
    """The whole shell, on the native Tigrex: it draws, and it draws SOMETHING."""
    if not TIGREX.exists():
        print("SKIP: %s is not here (game data, never committed — docs/ASSETS.md)"
              % TIGREX.name)
        return
    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.render.viewport import Viewport, scene_bounds

    scene = open_scene(TIGREX)
    bind = scene_bounds(scene)
    assert bind.radius > 0, bind
    with Viewport(ctx, (320, 240)) as vp:
        vp.set_scene(scene)
        # 🔴 framed on the POSED extent, not the bind one. A bind pose is a splayed T
        # whose centre is nowhere near the standing animal; framing on it puts the
        # subject in a corner. `Viewport.bounds` reports the pose.
        assert np.allclose(vp.camera.target, vp.bounds.center), \
            "the camera must frame what is on screen"
        assert not np.allclose(vp.bounds.center, bind.center), \
            "this PAC's posed and bind centres coincide — the test proves nothing"
        vp.draw()
        img = vp.target.read()

        # the background is uniform; anything else on screen is the subject or the grid.
        bg = np.array([int(round(c * 255)) for c in vp.background[:3]])
        lit = int((np.abs(img[..., :3].astype(int) - bg).max(axis=2) > 12).sum())
        assert lit > 500, "only %d pixels differ from the background — nothing drew" % lit

        # and with every layer off, nothing does. Proves the count above is OUR pixels
        # and not, say, a mis-cleared background.
        vp.show_ground = vp.show_axes = vp.show_points = vp.show_bounds = False
        vp.show_mesh = vp.show_skeleton = False
        vp.draw()
        empty = vp.target.read()
        blank = int((np.abs(empty[..., :3].astype(int) - bg).max(axis=2) > 12).sum())
        assert blank == 0, "%d pixels drew with every layer off" % blank
    print("viewport          %d bones / %d verts drew %d lit pixels; layers off -> 0"
          % (scene.rig.n_bones, scene.n_vertices, lit))


def test_draw_restores_the_framebuffer_binding(ctx):
    """🔴 `Viewport.draw` must leave the binding exactly as it found it.

    Binding an FBO is global GL state, and in the app `draw` runs INSIDE the host's
    frame. Leaving our offscreen target bound made imgui render the whole UI into it,
    so the window's back buffer was never drawn to — a completely black window, while
    every offscreen assertion still passed because the FBO did contain the right
    pixels. This is the assertion that fails instead.
    """
    if not TIGREX.exists():
        print("SKIP: no game data")
        return
    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.render.target import Target
    from mhfu_monster_editor.render.viewport import Viewport

    scene = open_scene(TIGREX)
    with Viewport(ctx, (120, 90)) as vp:
        vp.set_scene(scene)
        # A standalone context has NO default framebuffer (`ctx.screen` is None), so
        # the case that matters is the app's: some OTHER framebuffer is bound — the
        # host's — and it must still be bound afterwards.
        with Target(ctx, (32, 32), samples=0) as host:
            host.use()
            before = ctx.fbo
            assert before is not None, "the fixture did not bind a framebuffer"
            vp.draw()
            assert ctx.fbo is before, \
                "draw() left %s bound instead of the host's %s" % (ctx.fbo, before)
            # and the host can still draw into it: the viewport is restored too, not
            # left clipped to the viewport panel's rectangle.
            assert ctx.viewport == (0, 0, 32, 32), ctx.viewport
    print("state             draw() restores the framebuffer binding AND the viewport")


def test_the_reference_actor_stands_beside_the_port(ctx):
    """Issue #34: a SECOND monster in the same viewport, offset along the flank axis.

    Checked as pixels and as geometry, because "it drew something" and "it drew
    something in the right place" are different claims: the reference has to widen the
    framed bounds and light more of the picture, not overprint the first animal.
    """
    if not TIGREX.exists():
        print("SKIP: no game data")
        return
    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.render.viewport import Viewport

    scene = open_scene(TIGREX)
    with Viewport(ctx, (320, 240)) as vp:
        vp.set_scene(scene)
        alone = vp.bounds
        vp.draw()
        bg = np.array([int(round(c * 255)) for c in vp.background[:3]])
        lit_alone = int((np.abs(vp.target.read()[..., :3].astype(int) - bg)
                         .max(axis=2) > 12).sum())

        ref = vp.set_reference(scene)            # itself, which is enough to place it
        assert ref.offset[0] > 0 and ref.offset[1] == 0 and ref.offset[2] == 0, ref.offset
        assert ref.offset[0] > alone.radius, "the two would overlap"
        both = vp.bounds
        assert both.radius > alone.radius, (both.radius, alone.radius)
        assert both.hi[0] > alone.hi[0], "the union did not grow along the offset axis"

        vp.draw()
        lit_both = int((np.abs(vp.target.read()[..., :3].astype(int) - bg)
                        .max(axis=2) > 12).sum())
        assert lit_both > lit_alone, (lit_both, lit_alone)

        # its transport is its OWN: same rate, its own cursor and its own end.
        vp.play_reference_clip(scene.clips[0])
        assert ref.playback.end == scene.clips[0].frames
        vp.playback.speed = 3.0
        vp.playback.playing = True
        vp.tick(1.0)
        assert ref.playback.speed == 3.0, "the rate is the action's and is shared"
        assert ref.playback.phase > 0, "the reference did not advance"

        vp.clear_reference()
        assert vp.reference is None
        # back to the port alone — compared against the port's CURRENT pose, not the
        # one it had before `tick`, which moved it.
        assert np.allclose(vp.bounds.hi, vp.mesh.bounds.hi), "the union outlived the reference"
    print("reference         a second actor at +%.0f u widened the frame and lit "
          "%d -> %d pixels" % (ref.offset[0], lit_alone, lit_both))


def test_the_hurtbox_gizmos_ride_the_pose(ctx):
    """Issue #10: collision volumes on the LIVE posed skeleton.

    The claim worth testing is not "spheres appeared" — it is that a volume follows
    its bone's full matrix. A sphere placed at a bone-RELATIVE offset that only
    tracked the joint ORIGIN would look correct at bind and drift the moment the bone
    turned, which is exactly the fault a still picture cannot show.

    Also checks the orphan count, because a volume on a bone the rig does not have is
    drawn nowhere and would otherwise vanish silently — the failure mode of copying a
    48-joint host's bone indices onto a 46-joint port.
    """
    if not TIGREX.exists():
        print("SKIP: no game data")
        return
    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.render.hitboxes import DIM_ALPHA, Volume
    from mhfu_monster_editor.render.viewport import Viewport

    scene = open_scene(TIGREX)
    nb = scene.rig.n_bones
    vols = [Volume(bone=2, radius=97.0, part=1, hitzone_row=2, a=(0.0, -30.0, 30.0)),
            Volume(bone=6, radius=65.0, part=4, hitzone_row=5, a=(35.0, 0.0, 0.0),
                   b=(330.0, 0.0, 0.0)),                        # a capsule
            Volume(bone=nb + 5, radius=50.0, part=7)]           # off the rig
    with Viewport(ctx, (320, 240)) as vp:
        vp.set_scene(scene)
        bg = np.array([int(round(c * 255)) for c in vp.background[:3]])

        def lit():
            vp.draw()
            return int((np.abs(vp.target.read()[..., :3].astype(int) - bg)
                        .max(axis=2) > 12).sum())

        vp.show_mesh = vp.show_skeleton = vp.show_ground = False
        assert lit() == 0, "something drew with every layer off"

        ov = vp.set_hitboxes(vols)
        assert ov is not None and vp.show_hitboxes
        assert len(ov.orphans) == 1, ov.orphans
        assert len(ov.shown()) == 2, "the orphan was drawn anyway"
        assert lit() > 0, "the gizmos drew nothing"

        # ride the pose: advance a clip and the volumes move with their bones
        before = ov.world_centres().copy()
        vp.play_clip(scene.clips[0])
        vp.playback.playing = True
        vp.tick(0.5)
        after = ov.world_centres()
        assert not np.allclose(before, after), "the volumes did not follow the pose"

        # ... and they follow the bone's ROTATION, not just its origin. The offset
        # from the joint has to change direction as the joint turns.
        j0 = vp.skeleton.positions[2]
        assert not np.allclose(after[0] - j0, before[0] - j0, atol=1e-6), \
            "the offset stayed fixed in world space — the bone matrix was ignored"

        # part isolation: hiding a part removes it from the draw and from picking
        ov.set_visible_parts([1])
        assert [v.part for v in ov.shown()] == [1]
        lit_one_part = lit()
        ov.set_visible_parts(None)
        assert lit() > lit_one_part, "un-hiding a part did not draw more"

        # Selecting a part does two things at once, and they pull the pixel count in
        # OPPOSITE directions — the selection gains a filled shell while everything
        # else drops to 5%. Measured together they very nearly cancel, so each is
        # isolated instead.
        ov.set_visible_parts([1])                    # the selection, alone
        ov.set_selected_part(None)
        outline_only = lit()
        ov.set_selected_part(1)
        filled = lit()                               # draws, so the shell is built
        assert ov._fill.count > 0, "selecting a part did not build a shell"
        assert filled > outline_only, "the filled shell lit no extra pixels"

        ov.set_visible_parts([4])                    # a part that is NOT selected
        dimmed = lit()
        ov.set_selected_part(None)
        assert lit() > dimmed, "an unselected part was not dimmed"

        ov.set_visible_parts(None)
        ov.set_selected_part(4)
        lit()
        assert ov._fill.count > 0, "the shell did not follow the selection"
        ov.set_selected_part(None)
        lit()
        assert ov._fill.count == 0, "the shell outlived the selection"

        # The HOST actor beside the port carries the gizmos too, on ITS rig — the
        # comparison the side-by-side exists for. Attached in BOTH orders, because the
        # Parts panel and the Action panel get used in either.
        ov.set_selected_part(1)
        vp.set_reference_hitboxes(vols[:2])
        assert vp.reference is None, "no reference yet — this must not have made one"
        both = vp.set_reference(scene)                 # ... attached afterwards
        assert both.hitboxes is not None, "the reference did not pick the volumes up"
        assert both.hitboxes.selected_part == 1, "the focus did not follow"
        n_shown = len(both.hitboxes.shown())
        # measured against the SAME camera with the host's gizmos off, since
        # `set_reference` reframes on the union and that alone changes every count.
        with_ref = lit()
        vp.set_reference_hitboxes(None)
        assert vp.reference.hitboxes is None
        assert with_ref > lit(), "the reference's gizmos lit nothing"

        vp.clear_reference()
        vp.set_reference(scene)                        # attached FIRST this time
        vp.set_reference_hitboxes(vols[:2])
        assert vp.reference.hitboxes is not None, "the later volumes were dropped"
        assert vp.reference.hitboxes.selected_part == 1
        assert len(vp.reference.hitboxes.shown()) == n_shown, \
            "the two attach orders produced different geometry"

        vp.clear_hitboxes()
        assert vp.hitboxes is None and not vp.show_hitboxes
        assert vp.reference.hitboxes is None, "the reference kept its gizmos"
        assert lit() == 0, "the gizmos outlived clear_hitboxes"
        vp.clear_reference()
    print("hurtboxes         %d volume(s), 1 orphan; centres moved with the pose and "
          "with the bone's rotation; the selection fills, the rest dims to %d%%; "
          "the host actor carries them on its own rig"
          % (len(vols), round(DIM_ALPHA * 100)))


def test_render_to_file(ctx):
    """`render_to_file` opens its OWN context — the ``--headless`` path end to end."""
    if not TIGREX.exists():
        print("SKIP: no game data for the file render")
        return
    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.render.viewport import render_to_file

    scene = open_scene(TIGREX)
    with tempfile.TemporaryDirectory() as d:
        out = render_to_file(scene, Path(d) / "sub" / "shot.png", size=(200, 150),
                             view="side")
        assert out.exists() and out.stat().st_size > 1000, out
        try:
            from PIL import Image
        except ImportError:
            print("render_to_file    wrote %d bytes" % out.stat().st_size)
            return
        im = Image.open(out)
        assert im.size == (200, 150), im.size
    print("render_to_file    made its own context, wrote a %s PNG, tore it down"
          % (im.size,))


def main() -> int:
    test_png_encoders_agree()
    test_hurtbox_surfaces_are_exact()
    ctx = _context()
    if ctx is None:
        print("\ntest_render_headless: SKIPPED (no GL) — the camera maths is covered "
              "by test_render_camera.py")
        return 0
    try:
        test_target_reads_back_what_was_cleared(ctx)
        test_read_is_top_down(ctx)
        test_viewport_draws_the_scene(ctx)
        test_draw_restores_the_framebuffer_binding(ctx)
        test_the_reference_actor_stands_beside_the_port(ctx)
        test_the_hurtbox_gizmos_ride_the_pose(ctx)
        test_render_to_file(ctx)
    finally:
        ctx.release()
    print("\ntest_render_headless: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
