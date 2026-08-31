"""The app shell: a hello_imgui window, a docking layout, and the viewport in it.

    python -m mhfu_monster_editor workspace/extracted/data_files/file_06185.bin

🔴 **`hello_imgui` owns the window; moderngl only attaches to its context.**
`imgui-bundle` ships its *own* `libglfw`, and so does the `glfw` PyPI package that
`moderngl-window`'s GLFW backend imports. Loading both into one process on macOS
registers the GLFW Objective-C classes twice —

    objc: Class GLFWWindow is implemented in both .../glfw/libglfw.3.dylib and
    .../imgui_bundle/libglfw.3.dylib. ... may cause spurious casting failures and
    mysterious crashes.

— so this package does not depend on `moderngl-window` at all. `hello_imgui` gives the
window, input, DPI, docking and the layout `.ini` anyway, which is everything issue #5
wanted from it, and `moderngl.create_context()` binds to the context it made current.
One `libglfw` in the process, and no hand-written input plumbing.

The other rule this file follows: **the 3D view is drawn into an FBO and shown with
`imgui.image`**, never straight to the back buffer. That is what makes the windowed
picture and the `--headless` PNG the same pixels — see :mod:`..render.viewport`.
"""
from __future__ import annotations

from typing import Optional, Tuple

from ..render.camera import VIEWS
from ..render.context import ContextError, attached, describe
from ..render.mesh import MODES
from ..render.skeleton import undriven_geometry
from ..render.viewport import Viewport

#: where hello_imgui remembers window size, position and the docking layout.
INI_NAME = "mhfu_monster_editor"

_DRAG_ORBIT = 0     # imgui mouse button ids
_DRAG_PAN = 1
_DRAG_PAN_ALT = 2


def _viewport_button_flags() -> int:
    """An invisible button answers to the LEFT button only unless told otherwise, and
    the viewport needs all three — pan is on right and middle."""
    from imgui_bundle import imgui
    return (imgui.ButtonFlags_.mouse_button_left.value
            | imgui.ButtonFlags_.mouse_button_right.value
            | imgui.ButtonFlags_.mouse_button_middle.value)


class EditorApp:
    """The shell. Issues #6 and #7 add panels; this owns the window and the viewport.

    Held deliberately thin: everything that can be decided without a window lives in
    `core` or `render` and is tested there. What is left here is imgui wiring.
    """

    def __init__(self, scene, *, size: Tuple[int, int] = (1440, 900),
                 title: Optional[str] = None, view: str = "three",
                 startup=None) -> None:
        self.scene = scene
        #: the parsed CLI namespace, applied to the viewport on the first frame so the
        #: window opens showing what the same flags would have rendered headless.
        self.startup = startup
        self.size = size
        self.title = title or "mhfu_monster_editor — %s" % scene.name
        self.initial_view = view
        self.ctx = None
        self.viewport: Optional[Viewport] = None
        self.status = ""
        self._error: Optional[str] = None
        self._hovered = False
        #: draw each joint's index over the viewport. Off by default — 48 numbers on
        #: top of the animal is a lot, and it is the bone WORK that wants them.
        self.show_joint_ids = False
        #: joints that carry geometry no clip drives. → `render.skeleton`
        self.orphans = {}
        self._counts = None

    # ---- lifecycle ---------------------------------------------------- #
    def _ensure_gl(self) -> bool:
        """Attach on the first frame, when hello_imgui's context is current.

        Not in ``__init__``: there is no context yet at construction time, and
        `moderngl.create_context` would bind to whatever happened to be current.
        """
        if self.viewport is not None:
            return True
        if self._error:
            return False
        try:
            self.ctx = attached()
            self.viewport = Viewport(self.ctx, self.size)
            self.viewport.set_scene(self.scene)
            self.viewport.camera.look(self.initial_view)
            self.status = describe(self.ctx)
            self.orphans = undriven_geometry(self.scene)
            if self.startup is not None:
                from ..__main__ import apply_startup
                apply_startup(self.viewport, self.startup)
        except ContextError as e:
            self._error = str(e)
            return False
        return True

    # ---- panels ------------------------------------------------------- #
    def _viewport_panel(self) -> None:
        from imgui_bundle import imgui

        if self._error:
            imgui.text_wrapped(self._error)
            return
        if not self._ensure_gl():
            return

        avail = imgui.get_content_region_avail()
        w, h = int(max(avail.x, 1)), int(max(avail.y, 1))
        # The FBO is sized in FRAMEBUFFER pixels while imgui lays out in logical
        # points. Where those differ — a HiDPI display — rendering at point resolution
        # is the difference between a crisp viewport and a soft one.
        #
        # ⚠️ Trust imgui for this and NOT `ctx.screen.size`. moderngl reports the
        # attached context's screen as 2000x1400 for a window whose real default
        # framebuffer is 1000x700 — measured by clearing the whole thing and finding
        # only the bottom-left quarter had changed. Calibrating against that number
        # renders every frame at 4x the pixels it needs, and makes any read-back of
        # the back buffer run off the end of it into undefined memory (which reads as
        # black, and looks exactly like a broken renderer).
        scale = _framebuffer_scale()
        self.viewport.resize((int(w * scale), int(h * scale)))
        self.viewport.draw()

        pos = imgui.get_cursor_screen_pos()
        size = imgui.ImVec2(float(w), float(h))
        uv0, uv1 = self.viewport.target.uv
        imgui.image(imgui.ImTextureRef(self.viewport.target.gl_texture_id), size,
                    imgui.ImVec2(*uv0), imgui.ImVec2(*uv1))

        # 🔴 An `imgui.image` is NOT an interactive item: `is_item_active` is always
        # false on one, so gating a drag on it means the camera never moves. The
        # documented way is an invisible button laid over the picture — which also
        # gives real mouse CAPTURE, so a drag that leaves the panel keeps orbiting
        # instead of stopping at the edge.
        imgui.set_cursor_screen_pos(pos)
        imgui.invisible_button("##viewport", size, _viewport_button_flags())
        self._hovered = imgui.is_item_hovered()
        self._camera_input(h, active=imgui.is_item_active())
        self._joint_labels(imgui, pos, (w, h))
        _overlay_text(imgui, pos, self._hud())

    def _camera_input(self, view_h: int, *, active: bool) -> None:
        """Left-drag orbits, right/middle-drag pans, wheel dollies.

        ``active`` is the invisible button's — true from press until release, wherever
        the pointer has wandered to. The wheel is gated on HOVER instead, because a
        scroll needs no press to capture.
        """
        from imgui_bundle import imgui

        cam = self.viewport.camera
        if active:
            if imgui.is_mouse_dragging(_DRAG_ORBIT):
                d = imgui.get_mouse_drag_delta(_DRAG_ORBIT)
                cam.orbit(d.x, d.y)
                imgui.reset_mouse_drag_delta(_DRAG_ORBIT)
            for btn in (_DRAG_PAN, _DRAG_PAN_ALT):
                if imgui.is_mouse_dragging(btn):
                    d = imgui.get_mouse_drag_delta(btn)
                    cam.pan(d.x, d.y, view_h)
                    imgui.reset_mouse_drag_delta(btn)
        wheel = imgui.get_io().mouse_wheel
        if self._hovered and wheel:
            cam.dolly(wheel)

    def _hud(self) -> str:
        cam = self.viewport.camera
        return "%s\nyaw %.0f°  pitch %.0f°  dist %.0f" % (self.status, cam.yaw,
                                                          cam.pitch, cam.distance)

    def _scene_panel(self) -> None:
        from imgui_bundle import imgui

        sc = self.scene
        imgui.text(sc.name)
        imgui.text_disabled(sc.game)
        imgui.separator()
        for label, value in (("bones", sc.rig.n_bones),
                             ("groups", len(sc.groups)),
                             ("vertices", sc.n_vertices),
                             ("textures", len(sc.textures)),
                             ("clips", len(sc.clips))):
            imgui.text("%-9s %d" % (label, value))
        if sc.notes:
            imgui.separator()
            imgui.text_disabled("notes")
            for n in sc.notes:
                imgui.text_wrapped("• %s" % n)

    def _view_panel(self) -> None:
        from imgui_bundle import imgui

        if self.viewport is None:
            imgui.text_disabled("no GL context yet")
            return
        vp, cam = self.viewport, self.viewport.camera
        for i, name in enumerate(VIEWS):
            if i % 3:
                imgui.same_line()
            if imgui.button(name, imgui.ImVec2(72, 0)):
                cam.look(name)
        if imgui.button("frame", imgui.ImVec2(72, 0)):
            cam.frame(vp.bounds)

        imgui.separator()
        imgui.text_disabled("shading")
        changed, mode = imgui.combo("##mode", vp.mesh.mode, list(MODES))
        if changed:
            vp.mesh.mode = mode
        _, vp.show_mesh = imgui.checkbox("mesh", vp.show_mesh)
        imgui.same_line()
        _, vp.wireframe = imgui.checkbox("wire", vp.wireframe)

        imgui.separator()
        imgui.text_disabled("skeleton")
        _, vp.show_skeleton = imgui.checkbox("bones", vp.show_skeleton)
        imgui.same_line()
        _, vp.skeleton_xray = imgui.checkbox("x-ray", vp.skeleton_xray)
        _, self.show_joint_ids = imgui.checkbox("indices", self.show_joint_ids)

        imgui.separator()
        imgui.text_disabled("reference")
        _, vp.show_ground = imgui.checkbox("ground", vp.show_ground)
        imgui.same_line()
        _, vp.show_axes = imgui.checkbox("axes", vp.show_axes)
        _, vp.show_bounds = imgui.checkbox("bounds", vp.show_bounds)
        imgui.same_line()
        _, vp.show_points = imgui.checkbox("bind pts", vp.show_points)
        _, cam.fov = imgui.slider_float("fov", cam.fov, 15.0, 90.0)

    def _joints_panel(self) -> None:
        """The joint list: select, highlight, isolate. `MHFU_VIEW_HILITE`/`ONLY`."""
        from imgui_bundle import imgui

        vp = self.viewport
        if vp is None or vp.skeleton is None:
            imgui.text_disabled("no scene yet")
            return
        sk = vp.skeleton

        imgui.text("fork %d" % sk.fork)
        imgui.same_line()
        imgui.text_disabled("lead %s" % (list(sk.lead) or "-"))
        if self.orphans:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.70, 0.20, 1.0))
            imgui.text_wrapped(
                "⚠ %d vertices hang on joints no clip drives (%s) — they stay at bind, "
                "which is why they sit apart from the animal."
                % (sum(self.orphans.values()),
                   ", ".join(str(j) for j in sorted(self.orphans))))
            imgui.pop_style_color()

        changed, iso = imgui.combo("isolate", vp.mesh.isolate,
                                   ["off", "only tagged", "hide tagged"])
        if changed:
            vp.mesh.isolate = iso
        if imgui.button("clear tags"):
            vp.tag_joints(())
        imgui.same_line()
        if imgui.button("tag lead+fork"):
            vp.tag_joints(list(sk.lead))
        imgui.separator()

        tagged = set(vp.mesh.tagged)
        counts = self._joint_vertex_counts()
        if imgui.begin_child("##joints"):
            for j in range(len(sk.positions)):
                on = j in tagged
                hit, on = imgui.checkbox("##t%d" % j, on)
                if hit:
                    vp.tag_joints((tagged | {j}) if on else (tagged - {j}))
                imgui.same_line()
                label = "%2d  %s%s" % (j, "· " * 0, _joint_note(sk, j, counts.get(j, 0)))
                if imgui.selectable(label, sk.selected == j)[0]:
                    vp.select_joint(None if sk.selected == j else j)
            imgui.end_child()

    def _joint_labels(self, imgui, pos, size) -> None:
        """Joint indices drawn over the picture, and click-to-select on the joint.

        The numbers are 2D text on imgui's draw list rather than 3D geometry: they
        stay the same size at every zoom, they never need a font atlas in GL, and the
        projection they use is the very matrix the frame was drawn with — so a label
        cannot drift from its joint.
        """
        vp = self.viewport
        sk = vp.skeleton
        if sk is None:
            return
        mvp = vp.camera.mvp(vp.target.aspect)

        if self._hovered and imgui.is_mouse_clicked(_DRAG_ORBIT) \
                and not imgui.is_mouse_dragging(_DRAG_ORBIT):
            m = imgui.get_io().mouse_pos
            hit = sk.pick(mvp, size, m.x - pos.x, m.y - pos.y)
            if hit is not None:
                vp.select_joint(None if sk.selected == hit else hit)

        if not self.show_joint_ids and sk.selected is None:
            return
        got = sk.project(mvp, size)
        if got is None:
            return
        xy, ok = got
        draw = imgui.get_window_draw_list()
        plain = imgui.get_color_u32(imgui.ImVec4(0.80, 0.84, 0.92, 0.90))
        hot = imgui.get_color_u32(imgui.ImVec4(0.98, 0.30, 0.32, 1.0))
        for j in range(len(xy)):
            if not ok[j]:
                continue
            if not self.show_joint_ids and j != sk.selected:
                continue
            draw.add_text(imgui.ImVec2(pos.x + xy[j][0] + 6, pos.y + xy[j][1] - 7),
                          hot if j == sk.selected else plain, str(j))

    def _joint_vertex_counts(self):
        if self._counts is None:
            import numpy as np
            dom = self.scene.merged.dominant()
            self._counts = {int(j): int(n) for j, n in
                            zip(*np.unique(dom, return_counts=True)) if j >= 0}
        return self._counts

    # ---- run ---------------------------------------------------------- #
    def runner_params(self):
        """The hello_imgui description of this app — window, docking, panels."""
        from imgui_bundle import hello_imgui

        p = hello_imgui.RunnerParams()
        p.app_window_params.window_title = self.title
        p.app_window_params.window_geometry.size = self.size
        p.app_window_params.restore_previous_geometry = True
        # ⚠️ hello_imgui defaults to the CURRENT folder, which drops a settings file in
        # whatever directory the tool was launched from — the repo root, in practice.
        # The per-user config folder is the right place on all three platforms:
        # ~/Library/Application Support, ~/.config, %APPDATA%.
        p.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
        p.ini_filename = "%s.ini" % INI_NAME

        p.imgui_window_params.default_imgui_window_type = \
            hello_imgui.DefaultImGuiWindowType.provide_full_screen_dock_space
        p.imgui_window_params.enable_viewports = False
        p.imgui_window_params.show_menu_bar = True
        p.imgui_window_params.show_status_bar = True
        p.callbacks.show_status = lambda: _status(self)

        p.docking_params = _docking(self)
        return p

    def run(self) -> None:
        from imgui_bundle import hello_imgui

        try:
            hello_imgui.run(self.runner_params())
        finally:
            if self.viewport is not None:
                self.viewport.release()
                self.viewport = None


# --------------------------------------------------------------------------- #
# layout
# --------------------------------------------------------------------------- #
def _docking(app: EditorApp):
    """One split: a left dock for the inspectors, the rest for the viewport.

    Deliberately minimal — issue #5 asks for an *empty* docking layout, and #6/#7 add
    their windows to :attr:`dockable_windows` rather than reshaping this.
    """
    from imgui_bundle import hello_imgui, imgui

    split = hello_imgui.DockingSplit()
    split.initial_dock = "MainDockSpace"
    split.new_dock = "Left"
    split.direction = imgui.Dir.left
    split.ratio = 0.22

    def win(label, dock, fn, focus=False):
        w = hello_imgui.DockableWindow()
        w.label = label
        w.dock_space_name = dock
        w.gui_function = fn
        w.is_visible = True
        w.focus_window_at_next_frame = focus
        return w

    viewport = win("Viewport", "MainDockSpace", app._viewport_panel, focus=True)
    # the 3D view fills its panel; imgui padding would leave a border and, worse,
    # make the FBO and the panel disagree about size by a few pixels every frame.
    viewport.imgui_window_flags = _no_scroll_flags()

    d = hello_imgui.DockingParams()
    d.docking_splits = [split]
    d.dockable_windows = [viewport,
                          win("Scene", "Left", app._scene_panel),
                          win("View", "Left", app._view_panel),
                          win("Joints", "Left", app._joints_panel)]
    return d


def _no_scroll_flags() -> int:
    from imgui_bundle import imgui
    return (imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_scroll_with_mouse.value)


def _status(app: EditorApp) -> None:
    from imgui_bundle import imgui
    imgui.text(app.status or "attaching to the GL context…")


def _framebuffer_scale() -> float:
    """Framebuffer pixels per imgui point, as the backend reports it."""
    from imgui_bundle import imgui
    fb = imgui.get_io().display_framebuffer_scale
    return float(max(fb.x, 1.0))


def _joint_note(sk, j: int, verts: int) -> str:
    """The one-line role of a joint in the list: fork, lead chain, and its geometry."""
    bits = []
    if j == sk.fork:
        bits.append("FORK")
    elif j in sk.lead:
        bits.append("lead")
    if sk.driven is not None and j not in sk.driven:
        bits.append("undriven")
    if verts:
        bits.append("%dv" % verts)
    return " ".join(bits)


def _overlay_text(imgui, pos, text: str) -> None:
    """Draw the HUD over the image without a window that could steal the mouse."""
    draw = imgui.get_window_draw_list()
    col = imgui.get_color_u32(imgui.ImVec4(0.75, 0.78, 0.84, 0.85))
    for i, line in enumerate(text.splitlines()):
        draw.add_text(imgui.ImVec2(pos.x + 10, pos.y + 8 + i * 16), col, line)


def run(scene, **kw) -> None:
    """Open the editor on a scene. Blocks until the window closes."""
    EditorApp(scene, **kw).run()
