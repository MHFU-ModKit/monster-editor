"""The panels' PYTHON, without a window: every `*_panel` run over a real headless
viewport under a permissive `imgui` stand-in, through the same state script the
smoke test drives.

`test_ui_smoke.py` is the honest end-to-end check — a real window, the real back
buffer — and it needs a display AND a foreground process: launched from a background
shell on macOS the window never receives a frame, so it cannot run from an agent
session at all. What it would have caught in the panel code (a typo, a wrong
attribute, a None nobody guarded, a helper a refactor deleted) does not need a
window; it needs the panel functions CALLED with the app in every state they branch
on. That is what this does. The stub answers every imgui call with "nothing
happened" and the right shape of return value, so the code paths run and the
widgets' layout is not checked — that stays the smoke test's job.

Needs moderngl and a GL driver for the viewport (the panels read it); prints the
per-platform diagnosis and exits 0 without one, like `test_render_headless.py`.
"""
from __future__ import annotations

import os
import sys
import tempfile
import types
from pathlib import Path

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

TIGREX = _ROOT / "workspace" / "extracted" / "data_files" / "file_06185.bin"
ZINOGRE = _ROOT / "ports" / "zinogre.toml"


class _Vec:
    def __init__(self, *v):
        self.x = v[0] if v else 0.0
        self.y = v[1] if len(v) > 1 else 0.0
        self.z = v[2] if len(v) > 2 else 0.0
        self.w = v[3] if len(v) > 3 else 0.0


class _Enum:
    """`imgui.Col_.text`, `imgui.TableFlags_.row_bg.value`, ... — any name, value 0."""
    def __getattr__(self, name):
        return self

    @property
    def value(self):
        return 0

    def __or__(self, other):
        return 0

    def __ror__(self, other):
        return 0

    def __int__(self):
        return 0


class _Io:
    delta_time = 1.0 / 60.0
    mouse_wheel = 0.0
    mouse_pos = _Vec(0.0, 0.0)
    display_size = _Vec(1000.0, 700.0)
    display_framebuffer_scale = _Vec(1.0, 1.0)


class _DrawList:
    """`add_text`, `add_rect_filled`, `add_line`, ... — all no-ops."""
    def __getattr__(self, name):
        return lambda *a, **k: None


class FakeImgui:
    """Every call returns what a frame with no input would: buttons unpressed,
    widgets unchanged (`(False, value)`), tables open, tooltips never wanted."""
    calls = 0

    ImVec2 = _Vec
    ImVec4 = _Vec
    ImTextureRef = staticmethod(lambda tid: tid)
    Col_ = _Enum()
    TableFlags_ = _Enum()
    TableColumnFlags_ = _Enum()
    SelectableFlags_ = _Enum()
    ColorEditFlags_ = _Enum()
    Cond_ = _Enum()
    Dir = _Enum()
    WindowFlags_ = _Enum()
    ButtonFlags_ = _Enum()
    Key = _Enum()

    def __getattr__(self, name):
        # anything not named explicitly below: a no-op returning None/False
        FakeImgui.calls += 1
        return lambda *a, **k: False

    # -- widgets that return (changed, value) -------------------------------- #
    def checkbox(self, label, v):
        return False, v

    def radio_button(self, label, v):
        return False

    def input_int(self, label, v, *a):
        return False, v

    def input_float(self, label, v, *a):
        return False, v

    def input_float3(self, label, v, *a):
        return False, v

    def drag_float(self, label, v, *a):
        return False, v

    def slider_float(self, label, v, *a):
        return False, v

    def slider_int(self, label, v, *a):
        return False, v

    def combo(self, label, idx, items, *a):
        return False, idx

    def input_text(self, label, s, *a):
        return False, s

    def selectable(self, label, sel, *a):
        return False, sel

    def collapsing_header(self, label, *a):
        return True

    def tree_node(self, label, *a):
        return False

    def begin_table(self, *a):
        return True

    def begin_child(self, *a):
        return True

    def begin_combo(self, *a):
        return False

    def begin_popup(self, *a):
        return False

    def begin_tab_bar(self, *a):
        return True

    def begin_tab_item(self, *a):
        return True, True

    def get_io(self):
        return _Io()

    def get_content_region_avail(self):
        return _Vec(640.0, 400.0)

    def get_cursor_screen_pos(self):
        return _Vec(0.0, 0.0)

    def get_window_draw_list(self):
        return _DrawList()

    def get_color_u32(self, *a):
        return 0

    def calc_text_size(self, *a):
        return _Vec(50.0, 14.0)

    def get_frame_height(self):
        return 20.0

    def get_text_line_height(self):
        return 14.0

    def get_item_rect_min(self):
        return _Vec(0.0, 0.0)

    def get_item_rect_max(self):
        return _Vec(10.0, 10.0)

    def get_mouse_drag_delta(self, *a):
        return _Vec(0.0, 0.0)

    def is_item_hovered(self, *a):
        return False

    def is_item_active(self, *a):
        return False

    def is_mouse_clicked(self, *a):
        return False

    def is_mouse_dragging(self, *a):
        return False


def _install_fake_imgui():
    fake = types.ModuleType("imgui_bundle")
    fake.imgui = FakeImgui()
    hi = types.ModuleType("hello_imgui")
    fake.hello_imgui = hi
    sys.modules["imgui_bundle"] = fake
    sys.modules["imgui_bundle.imgui"] = fake.imgui           # type: ignore
    return fake


def _context():
    try:
        from mhfu_monster_editor.render.context import ContextError, describe, headless
    except ImportError as e:
        print("SKIP: moderngl is not installed (%s)" % e)
        return None
    try:
        return headless()
    except ContextError as e:
        print("SKIP: no GL —", describe(e) if callable(describe) else e)
        return None


def run_panels(ctx) -> dict:
    """Open the app object on the Tigrex, hand it a headless viewport, attach the
    Zinogre manifest, then call every panel through the smoke test's state script."""
    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.manifest import load
    from mhfu_monster_editor.render.viewport import Viewport
    from mhfu_monster_editor.ui.app import EditorApp

    scene = open_scene(TIGREX)
    app = EditorApp(scene, size=(1000, 700), ini_folder=tempfile.mkdtemp())
    app.ctx = ctx
    app.viewport = Viewport(ctx, (640, 400))
    app.viewport.set_scene(scene)
    app.scene.attach_manifest(load(ZINOGRE))
    panels = {name: getattr(app, name) for name in dir(app)
              if name.endswith("_panel") and callable(getattr(app, name))
              and name != "_viewport_panel"}        # the viewport needs hello_imgui
    got = {"frames": 0, "errors": [], "panels": sorted(panels)}

    def frame(step):
        try:
            step()
            for name, fn in panels.items():
                fn()
            app.viewport.tick(1.0 / 60.0)
            app.viewport.draw()
        except Exception as e:                            # noqa: BLE001
            import traceback
            got["errors"].append("frame %d: %s: %s\n%s" % (
                got["frames"], type(e).__name__, e, traceback.format_exc()))
        got["frames"] += 1

    def s_parts_on():
        app.show_parts = True
        app.sync_hitboxes()

    def s_part_sel():
        app.selected_part = 1
        if app.viewport.hitboxes is not None:
            app.viewport.hitboxes.set_selected_part(1)

    def s_host():
        app.show_host = True
        app.sync_reference()

    def s_adopt_parts():
        sess, host = app.part_session, app.host_parts()
        if sess is not None and host is not None:
            if host.has_grid:
                sess.adopt_grid(host.states)
            sess.adopt_volumes(host.spheres()[:8])
            app.parts_source = "port"
            app.sync_hitboxes()

    def s_edit_part():
        sess = app.part_session
        if sess is not None and sess.volumes():
            app.select_volume(0)
            sess.edit_volume(0, shape="capsule", to=[0.0, 0.0, 100.0])
            app.sync_hitboxes()

    # -- the attack side (#33) ---------------------------------------------- #
    def s_attacks_on():
        app.show_attacks = True
        app.sync_attacks()

    def s_pair():
        if app.intel is not None and app.intel.pair(1, 4) is not None:
            app.select_pair(1, 4)
        app.sync_attacks()

    def s_set():
        app.select_set(2)

    def s_adopt_set():
        sess, host = app.attack_session, app.host_attacks()
        if sess is not None and host is not None and host.set(2) is not None:
            sess.adopt_set(2, host.set(2).spheres, source="panel test")
            app.attacks_source = "port"
            app.sync_attacks()

    def s_edit_attack():
        sess = app.attack_session
        if sess is not None and sess.volumes():
            app.select_attack_volume(0)
            sess.edit_volume(0, shape="capsule", to=[0.0, 0.0, 100.0])
            sess.scale_volume(0, 2.0)
            sess.set_attack(6, power=40)
            app.sync_attacks()

    def s_keep_only():
        sess = app.attack_session
        if sess is not None and sess.volumes():
            sess.keep_only(0)
            app.select_attack_volume(0)
            app.sync_attacks()

    def s_unset():
        app.select_set(None)          # the "all authored sets" branch of the editor
        app._sets_of_move_only = False

    def s_host_source():
        app.attacks_source = "host"   # the read-only levers table
        app.select_set(2)
        app.sync_attacks()

    def s_browse():
        # browsing another overlay: every host-only branch must step aside
        app.browse_species(7)

    for step in (lambda: None, s_parts_on, s_part_sel, s_host, s_adopt_parts, s_edit_part,
                 s_attacks_on, s_pair, s_set, s_adopt_set, s_edit_attack, s_keep_only,
                 s_unset, s_host_source, s_browse):
        frame(step)
    got["status"] = app.status
    got["attack_overlay"] = app.viewport.attacks is not None
    got["ref_attacks"] = (app.viewport.reference is not None
                          and app.viewport.reference.attacks is not None)
    got["staged"] = 0 if app._attacks is None else app._attacks.pending
    app.viewport.release()
    return got


def test_every_panel_runs_through_the_attack_states():
    if not TIGREX.exists() or not ZINOGRE.exists():
        print("SKIP: no game data (docs/ASSETS.md)")
        return
    _install_fake_imgui()
    ctx = _context()
    if ctx is None:
        return
    try:
        got = run_panels(ctx)
    finally:
        ctx.release()
    assert not got["errors"], "\n\n".join(got["errors"])
    assert "_hitboxes_panel" in got["panels"], got["panels"]
    assert "_parts_panel" in got["panels"] and "_action_panel" in got["panels"]
    assert got["frames"] == 15
    assert got["attack_overlay"], "the attack overlay was never attached"
    assert got["ref_attacks"], "the reference did not carry the host's sets"
    assert got["staged"] > 0, "the staged session was lost between frames"
    print("panels            %d panels x %d states, %d imgui calls, no exception; "
          "status: %s" % (len(got["panels"]), got["frames"], FakeImgui.calls,
                          got["status"][:70]))


def main() -> int:
    test_every_panel_runs_through_the_attack_states()
    print("\ntest_ui_panels: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
