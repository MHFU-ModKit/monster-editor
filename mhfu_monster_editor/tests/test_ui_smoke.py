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


def run_smoke(pac: Path, frames: int = FRAMES, out: Path = None,
              ini_folder: str = None) -> dict:
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
    # ⚠️ a temp ini folder: the app remembers window size and docking layout per user,
    # and a smoke run must not overwrite a layout somebody arranged by hand.
    app = EditorApp(scene, size=(1000, 700), ini_folder=ini_folder or tempfile.mkdtemp())
    got = {"frames": 0, "error": None, "status": "", "fbo": None,
           "screen": None, "lit": 0}

    # every panel in the docking layout, not a hand-kept subset: a panel added by a
    # later issue that throws would otherwise be swallowed by the C++ runner and the
    # smoke test would still pass.
    panels = {name: getattr(app, name) for name in dir(app)
              if name.endswith("_panel") and callable(getattr(app, name))}

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

    got["panels"] = sorted(panels)
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

    # Drive the states a default-state run never reaches. The parts panel (#10) has
    # three branches nobody sees on frame 1 — the port source with a staged session,
    # an isolated part, and the EDITABLE grid — and each of them is a different pile
    # of imgui calls. A smoke test that only ever exercises the opening state would
    # have missed the panel that deleted its own helper methods in #8.
    def exercise():
        f = got["frames"]
        if f == 1 and app.scene.manifest is None:
            # the smoke scene is a bare host PAC, so the parts panel's port branch —
            # a staged session, an EDITABLE grid, the save row — is unreachable.
            # Attach a shipped manifest so those imgui calls actually run. Nothing
            # here saves, so no file is touched.
            man = _ROOT / "ports" / "zinogre.toml"
            if man.exists():
                from mhfu_monster_editor.manifest import load as _load
                app.scene.attach_manifest(_load(man))
                app._parts = None
        if f == 2:
            app.show_parts = True
            app.sync_hitboxes()
        elif f == 4:
            app.selected_part = 1
            if app.viewport is not None and app.viewport.hitboxes is not None:
                app.viewport.hitboxes.set_selected_part(1)
        elif f == 5:
            # the host actor beside the port, WITH the gizmos already on: the
            # branch where both overlays are alive at once.
            app.show_host = True
            app.sync_reference()
        elif f == 6:
            sess = app.part_session
            host = app.host_parts()
            if sess is not None and host is not None:
                if host.has_grid:
                    sess.adopt_grid(host.states)
                sess.adopt_volumes(host.spheres()[:8])
                app.parts_source = "port"
                app.sync_hitboxes()

    _counted = counted

    def counted():                                            # noqa: F811
        exercise()
        _counted()

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
        print("ui                %d frames, %d panels drew, viewport FBO %dx%d, back "
              "buffer %dx%d with %.0f%% lit, %s"
              % (got["frames"], len(got["panels"]), got["fbo"][0], got["fbo"][1], w, h,
                 100.0 * got["lit"] / (w * h), got["status"]))


def test_playback_suppresses_the_idle_throttle():
    """🔴 hello_imgui idles to 9 fps three seconds after the last input event.

    Right for a static pose, wrong for playback — which animates precisely when
    nobody is touching anything. The symptom is the frame rate collapsing a few
    seconds into a loop and recovering the moment the mouse moves.

    Measured rather than asserted on the flag alone: the app runs unattended for
    longer than `time_active_after_last_event`, and the frame rate over the LAST
    stretch has to stay well above `fps_idle`.
    """
    if not os.environ.get("MHFU_UI_SMOKE"):
        print("SKIP: set MHFU_UI_SMOKE=1 (needs a display)")
        return
    if not TIGREX.exists():
        print("SKIP: no game data")
        return
    import time

    from imgui_bundle import hello_imgui

    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.ui import EditorApp

    scene = open_scene(TIGREX)
    app = EditorApp(scene, size=(700, 500), ini_folder=tempfile.mkdtemp())
    idle_fps = hello_imgui.RunnerParams().fps_idling.fps_idle
    hold = hello_imgui.RunnerParams().fps_idling.time_active_after_last_event
    run_for = hold + 2.5
    got = {"n": 0, "late": 0, "t0": None, "mark": None, "idling": None, "err": None}

    real = app._viewport_panel

    def wrapped():
        try:
            if got["n"] == 0:
                real()
                # start playing on the first frame, then never touch the input again.
                app.viewport.playback.loop = True
                app.viewport.playback.play()
                got["t0"] = time.perf_counter()
            else:
                real()
        except BaseException as e:                           # noqa: BLE001
            import traceback
            got["err"] = "%s: %s\n%s" % (type(e).__name__, e, traceback.format_exc())
            hello_imgui.get_runner_params().app_shall_exit = True
            return
        got["n"] += 1
        now = time.perf_counter()
        if now - got["t0"] >= hold and got["mark"] is None:
            got["mark"] = (now, got["n"])          # start counting AFTER the idle kicks in
        if got["mark"] is not None:
            got["late"] = got["n"] - got["mark"][1]
        if now - got["t0"] >= run_for:
            got["idling"] = bool(
                hello_imgui.get_runner_params().fps_idling.enable_idling)
            got["elapsed"] = now - got["mark"][0]
            hello_imgui.get_runner_params().app_shall_exit = True

    app._viewport_panel = wrapped
    app.run()

    assert got["err"] is None, got["err"]
    assert got["mark"] is not None, "the app exited before the idle window elapsed"
    fps = got["late"] / max(got["elapsed"], 1e-6)
    assert got["idling"] is False, \
        "idling was left ENABLED while the transport was playing"
    assert fps > idle_fps * 2, (
        "only %.1f fps over the %.1f s AFTER the %.0f s idle window — the throttle "
        "engaged during playback (idle is %.0f fps)"
        % (fps, got["elapsed"], hold, idle_fps))
    print("idling            %.0f fps sustained %.1f s after the %.0f s idle window "
          "(idle would be %.0f)" % (fps, got["elapsed"], hold, idle_fps))


def main() -> int:
    test_app_opens_and_draws()
    test_playback_suppresses_the_idle_throttle()
    print("\ntest_ui_smoke: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
