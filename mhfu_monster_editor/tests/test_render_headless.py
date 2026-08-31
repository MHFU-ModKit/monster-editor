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
        test_render_to_file(ctx)
    finally:
        ctx.release()
    print("\ntest_render_headless: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
