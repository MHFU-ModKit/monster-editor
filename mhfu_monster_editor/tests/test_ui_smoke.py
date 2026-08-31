"""Does the app actually open? Opt-in, because it needs a display and takes focus.

    MHFU_UI_SMOKE=1 venv/bin/python mhfu_monster_editor/tests/test_ui_smoke.py

Without the variable it prints why it did nothing and exits 0, so it is safe in the
`for t in tests/test_*.py` loop and on the headless server.

It is worth having despite that: the thing most likely to break as issues #6 and #7 add
panels is not the maths, it is the window — an imgui-bundle API that moved, a docking
field that was renamed, a moderngl object built before the context was current. This
opens the real window, runs a few real frames with every panel drawing, saves what the
viewport produced, and closes. Any exception inside a panel is caught and re-raised
here rather than being swallowed by the C++ runner.

⚠️ macOS requires a window on the MAIN thread, so this cannot be run from a worker.
"""
from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

TIGREX = _ROOT / "workspace" / "extracted" / "data_files" / "file_06185.bin"
FRAMES = 10


def run_smoke(pac: Path, frames: int = FRAMES, out: Path = None) -> dict:
    """Open the app on ``pac``, draw ``frames`` frames, close. Returns what happened.

    🔴 Samples the **real back buffer** in `before_swap`, not our offscreen target.
    Checking the target is checking the wrong surface: it was full of the right pixels
    for the whole time the window was rendering completely black, because
    `Viewport.draw` had left our FBO bound and imgui was drawing the UI into it. The
    only honest end-to-end assertion is "what would actually be presented".
    """
    import numpy as np
    from imgui_bundle import hello_imgui

    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.ui import EditorApp

    scene = open_scene(pac)
    app = EditorApp(scene, size=(1000, 700))
    got = {"frames": 0, "error": None, "status": "", "fbo": None,
           "screen": None, "lit": 0}

    panels = {name: getattr(app, name) for name in
              ("_viewport_panel", "_scene_panel", "_view_panel")}

    def guard(name, fn):
        def wrapped():
            if got["error"]:
                return
            try:
                fn()
            except BaseException as e:                       # noqa: BLE001
                import traceback
                got["error"] = "%s in %s: %s\n%s" % (type(e).__name__, name, e,
                                                     traceback.format_exc())
                hello_imgui.get_runner_params().app_shall_exit = True
        return wrapped

    for name, fn in panels.items():
        setattr(app, name, guard(name, fn))

    # count on the viewport, which is the panel that has to reach GL.
    viewport = getattr(app, "_viewport_panel")

    def counted():
        viewport()
        if got["error"]:
            return
        got["frames"] += 1
        if got["frames"] >= frames:
            got["status"] = app.status
            if app.viewport is not None:
                got["fbo"] = app.viewport.target.size
            # ⚠️ `out` is the WINDOW, written by `before_swap` below — never the
            # offscreen target. Saving the target here as well would overwrite the one
            # picture that can show whether the app is visible with the one picture
            # that looks right even when it is not.
            hello_imgui.get_runner_params().app_shall_exit = True

    app._viewport_panel = counted

    def before_swap():
        """The finished frame, exactly as it is about to be presented.

        ⚠️ The size comes from imgui, NOT from `ctx.screen.size`. moderngl reports the
        attached context's screen as 2000x1400 for a window whose real default
        framebuffer is 1000x700 (measured: clearing the whole thing reddened only the
        bottom-left quarter). Reading at moderngl's figure runs off the end of the
        buffer, and the out-of-bounds three quarters come back BLACK — which is
        indistinguishable from the very bug this test exists to catch.
        """
        from imgui_bundle import imgui as _imgui

        if got["screen"] is not None or app.ctx is None or got["frames"] < frames - 1:
            return
        screen = app.ctx.screen
        if screen is None:                                   # pragma: no cover
            return
        io = _imgui.get_io()
        scale = max(io.display_framebuffer_scale.x, 1.0)
        w = int(io.display_size.x * scale)
        h = int(io.display_size.y * scale)
        if w < 1 or h < 1:                                   # pragma: no cover
            return
        img = np.frombuffer(screen.read(viewport=(0, 0, w, h), components=3,
                                        alignment=1),
                            dtype=np.uint8).reshape(h, w, 3)[::-1]
        got["screen"] = (w, h)
        # anything at all that is not the window's clear colour.
        got["lit"] = int((img.max(axis=2) > 24).sum())
        if out is not None:
            from mhfu_monster_editor.render.target import write_png
            write_png(out, np.concatenate(
                [img, np.full((h, w, 1), 255, np.uint8)], axis=2))

    real_params = app.runner_params

    def with_capture():
        p = real_params()
        p.callbacks.before_swap = before_swap
        return p

    app.runner_params = with_capture
    app.run()
    return got


def test_app_opens_and_draws():
    if not os.environ.get("MHFU_UI_SMOKE"):
        print("SKIP: set MHFU_UI_SMOKE=1 to open a real window (needs a display, and "
              "on macOS the main thread)")
        return
    if not TIGREX.exists():
        print("SKIP: %s is not here (game data, never committed — docs/ASSETS.md)"
              % TIGREX.name)
        return
    try:
        import imgui_bundle  # noqa: F401
    except ImportError as e:
        print("SKIP: imgui-bundle is not installed (%s)" % e)
        return

    with tempfile.TemporaryDirectory() as d:
        shot = Path(d) / "window.png"
        got = run_smoke(TIGREX, out=shot)
        assert got["error"] is None, got["error"]
        assert got["frames"] >= FRAMES, "the window closed after %d frames" % got["frames"]
        assert got["fbo"] and min(got["fbo"]) > 0, got["fbo"]
        assert shot.exists() and shot.stat().st_size > 1000, "nothing was saved"

        # 🔴 THE assertion. A black window is not a rendering nicety that went wrong;
        # it is the whole app being invisible, and it passed every offscreen check.
        assert got["screen"], "the back buffer was never sampled"
        w, h = got["screen"]
        assert got["lit"] > w * h * 0.02, (
            "the window presented %d lit pixels of %d — it is BLACK. Something left a "
            "framebuffer bound and imgui drew into it instead of the back buffer."
            % (got["lit"], w * h))
        print("ui                %d frames, viewport FBO %dx%d, back buffer %dx%d with "
              "%.0f%% lit, %s"
              % (got["frames"], got["fbo"][0], got["fbo"][1], w, h,
                 100.0 * got["lit"] / (w * h), got["status"]))


def main() -> int:
    test_app_opens_and_draws()
    print("\ntest_ui_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
