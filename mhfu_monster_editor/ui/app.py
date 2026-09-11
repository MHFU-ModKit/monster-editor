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

from pathlib import Path
from typing import Optional, Tuple

from ..render.camera import VIEWS
from ..render.context import ContextError, attached, describe
from ..render.mesh import MODES
from ..render.playback import GAME_HZ, OBSERVED_SPEEDS, root_travel, wall_clock
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
                 startup=None, ini_folder: Optional[str] = None) -> None:
        self.scene = scene
        #: the parsed CLI namespace, applied to the viewport on the first frame so the
        #: window opens showing what the same flags would have rendered headless.
        self.startup = startup
        self.size = size
        self.title = title or "mhfu_monster_editor — %s" % scene.name
        self.initial_view = view
        #: where the layout `.ini` lives. None = the per-user config folder. A TEST
        #: passes a temp directory: a smoke run must not overwrite the window size and
        #: docking layout somebody arranged by hand.
        self.ini_folder = ini_folder
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
        #: `align.Marker`s drawn on the timeline's marker strip — the HOST action's
        #: expectations in the CLIP's frame space. Filled by `_recompute_alignment`.
        self.markers = []
        #: the current `align.Alignment`, or None. → #9
        self.alignment = None
        #: `species/emNN.json` per species, loaded on demand
        self._intel_cache = {}
        #: which overlay's action table the panel shows. None = the manifest's host.
        self._browse = None
        self._hosts = None
        #: draw the HOST monster beside the port, playing the selected action's own
        #: clip. Off by default: a second PAC parse and a second skinned mesh. → #34
        self.show_host = False
        self._host_scene = {}
        self._host_clip = None
        #: the `(main, sub)` under inspection — a declared move's, or one being tried
        self._pair = None
        #: the Moves tab: the behaviour pairs as a graph of hand-offs (ui/graph.py)
        from .graph import MoveGraph
        self._graph = MoveGraph()
        self._move = None
        self._pair_filter = ""
        self._bind_buf = ""
        self._clip_filter = ""
        self._travel_cache = {}
        #: the part system (#10): which volumes to draw, and which part is isolated.
        self.show_parts = False
        #: "host" = the host overlay's own volumes, "port" = this manifest's.
        self.parts_source = "host"
        self.selected_part: Optional[int] = None
        #: ONE volume singled out for editing — an index into the port's list
        self.selected_volume: Optional[int] = None
        self._only_selected_part = False
        self._parts = None
        self._orphans = ()
        self._part_name_buf = ""
        self._grid_state = 0
        #: the last runtime module the Parts panel exported, for the deploy button
        self._hit_export = None
        #: the attack system (#33): the sets a move hits with, on the port's rig
        self.show_attacks = False
        #: "host" = the host overlay's own sets, "port" = this manifest's [[hitbox]]
        self.attacks_source = "host"
        self.selected_set: Optional[int] = None
        #: ONE attack volume singled out — an index into `AttackSession.volumes()`
        self.selected_attack_volume: Optional[int] = None
        #: list only the sets the selected pair/move hits with, or every set
        self._sets_of_move_only = True
        self._attacks = None
        self._attack_orphans = ()
        self._attack_label_buf = ""
        #: `clips.Coverage` + `LabelTrack`s + the build id, computed once. → #8
        self._vocab = None
        #: `clips.LabelSession` — manifest edits typed here, not yet on disk
        self._session = None
        self._edit_slot = None
        self._name_buf = ""
        self._label_buf = ""
        self._saved = ""
        #: the user's "Enable idling" setting, borrowed while a clip plays.
        self._idle_pref = None

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

    # ---- power saving ------------------------------------------------- #
    def _sync_idling(self) -> None:
        """Hold hello_imgui's idle off while a clip is playing, and hand it back after.

        🔴 hello_imgui idles the app to save power: `fps_idling.fps_idle` is **9**, and
        it engages `time_active_after_last_event` = **3 seconds** after the last input
        event. That is right for a static pose — a viewport nobody is touching should
        not burn a core — and wrong for playback, which is animating precisely when
        nobody is touching anything. The symptom is the frame rate collapsing a few
        seconds into a loop and recovering the instant the mouse moves.

        The clip still advances at the correct RATE while idling, because
        :meth:`Playback.advance` works in real seconds — 9 fps just means ~3.3 game
        frames are stepped per rendered frame, so the animation is *choppy*, not slow.

        Idling is BORROWED, not overridden: the setting the user had (there is a
        checkbox for it in the status bar) is captured on the first playing frame and
        restored when playback stops, so toggling it while paused still sticks.
        """
        from imgui_bundle import hello_imgui

        playing = self.viewport is not None and self.viewport.playback.playing
        idling = hello_imgui.get_runner_params().fps_idling
        if playing:
            if self._idle_pref is None:
                self._idle_pref = bool(idling.enable_idling)
            idling.enable_idling = False
        elif self._idle_pref is not None:
            idling.enable_idling = self._idle_pref
            self._idle_pref = None

    # ---- panels ------------------------------------------------------- #
    def _viewport_panel(self) -> None:
        from imgui_bundle import imgui

        if self._error:
            imgui.text_wrapped(self._error)
            return
        if not self._ensure_gl():
            return

        # the transport is advanced HERE, once per frame, from imgui's own delta —
        # not in the timeline panel, which is dockable and may be closed.
        self._sync_idling()
        self.viewport.tick(imgui.get_io().delta_time)

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
                "! %d vertices hang on joints no clip drives (%s) — they stay at bind, "
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
        # ⚠️ ImGui 1.90+ requires EndChild for EVERY BeginChild, return value or not —
        # skipping it on a collapsed or fully-clipped panel trips an assertion.
        imgui.begin_child("##joints")
        for j in range(len(sk.positions)):
            on = j in tagged
            hit, on = imgui.checkbox("##t%d" % j, on)
            if hit:
                vp.tag_joints((tagged | {j}) if on else (tagged - {j}))
            imgui.same_line()
            if imgui.selectable("%2d  %s" % (j, _joint_note(sk, j, counts.get(j, 0))),
                                sk.selected == j)[0]:
                vp.select_joint(None if sk.selected == j else j)
        imgui.end_child()


    # ---- the part system (issue #10) ---------------------------------- #
    @property
    def part_session(self):
        """Staged part edits. Lives in `parts.PartSession` for the same reason the
        label session does: the write path is testable without a window."""
        if self._parts is None and self.scene.manifest is not None:
            from ..parts import PartSession
            self._parts = PartSession(self.scene.manifest, self.scene.rig.n_bones)
        return self._parts

    def host_parts(self):
        """The HOST species' part intel, or None.

        Deliberately the host and not `browsing_species`: browsing another overlay's
        action table is a comparison (#9), but the volumes and the grid are what this
        port will actually ride, and drawing another species' spheres on the animal
        would be a picture of something that is not going to happen.
        """
        sp = self.host_species
        if sp is None:
            return None
        from ..intel import find_intel
        if sp not in self._intel_cache:
            self._intel_cache[sp] = find_intel(sp)
        si = self._intel_cache[sp]
        pt = getattr(si, "parts", None)
        return pt if (pt is not None and pt.present) else None

    def sync_hitboxes(self) -> None:
        """Push the chosen source's volumes into the viewport."""
        vp = self.viewport
        if vp is None or vp.scene is None:
            return
        if not self.show_parts:
            vp.clear_hitboxes()          # clears the reference's too
            self._orphans = ()
            return
        host = self.host_parts()
        if self.parts_source == "port":
            sess = self.part_session
            if sess is not None and host is not None:
                sess.capacity = host.capacity
            vols = sess.volumes() if sess is not None else list(
                getattr(self.scene.manifest, "hurtboxes", []) or [])
        else:
            vols = host.spheres() if host is not None else []
        ov = vp.set_hitboxes(vols)
        self._orphans = () if ov is None else ov.orphans
        if ov is not None:
            ov.set_selected_part(self.selected_part)
            ov.set_selected_volume(self.selected_volume
                                   if self.parts_source == "port" else None)
        # the HOST actor beside the port draws the HOST's own volumes, whatever the
        # port has authored — those bone indices were written for that rig, and the
        # side-by-side is what makes a sphere on the wrong joint legible.
        vp.set_reference_hitboxes([] if host is None else host.spheres())

    def select_volume(self, index: Optional[int]) -> None:
        """Single out one of the port's volumes, in the list and in the viewport."""
        self.selected_volume = index
        vp = self.viewport
        if vp is not None and vp.hitboxes is not None:
            vp.hitboxes.set_selected_volume(index if self.parts_source == "port"
                                            else None)

    # ---- the attack system (issue #33) -------------------------------- #
    @property
    def attack_session(self):
        """Staged [[hitbox]] / [[attack]] edits — `attacks.AttackSession`, testable
        without a window like the other two sessions."""
        if self._attacks is None and self.scene.manifest is not None:
            from ..attacks import AttackSession
            self._attacks = AttackSession(self.scene.manifest, self.scene.rig.n_bones)
            host = self.host_attacks()
            if host is not None:
                self._attacks.capacities = {st.index: st.capacity for st in host.sets}
        return self._attacks

    def host_attacks(self):
        """The HOST species' attack intel, or None. The host and not
        `browsing_species`, for the reason `host_parts` gives: these are the sets the
        port will actually hit with."""
        sp = self.host_species
        if sp is None:
            return None
        from ..intel import find_intel
        if sp not in self._intel_cache:
            self._intel_cache[sp] = find_intel(sp)
        si = self._intel_cache[sp]
        at = getattr(si, "attacks", None)
        return at if (at is not None and at.present) else None

    def host_intel(self):
        """The HOST's `SpeciesIntel` — not the overlay the Action panel may be
        browsing. A pair from another overlay has no attack ids in this table."""
        sp = self.host_species
        if sp is None:
            return None
        from ..intel import find_intel
        if sp not in self._intel_cache:
            self._intel_cache[sp] = find_intel(sp)
        return self._intel_cache[sp]

    def host_pair(self):
        """The selected pair's intel, when it is the host's; None while browsing."""
        if self._pair is None or not self.browsing_the_host:
            return None
        si = self.host_intel()
        return None if si is None else si.pair(*self._pair)

    def pair_sets(self) -> list:
        """The volume sets the selected pair's handler hits with (empty = none
        known, no pair selected, or the pair is another overlay's)."""
        host = self.host_attacks()
        p = self.host_pair()
        if host is None or p is None or not p.attack_ids:
            return []
        return host.sets_for(p.attack_ids, self.host_species)

    def sync_attacks(self) -> None:
        """Push the chosen source's attack volumes into the viewport, filtered to
        the selected set (or the selected move's sets) so 56 sets are not a fog."""
        vp = self.viewport
        if vp is None or vp.scene is None:
            return
        if not self.show_attacks:
            vp.clear_attacks()
            self._attack_orphans = ()
            return
        host = self.host_attacks()
        from ..render.hitboxes import attack_volumes_from
        if self.attacks_source == "port":
            sess = self.attack_session
            vols = sess.volumes() if sess is not None else list(
                getattr(self.scene.manifest, "hitboxes", []) or [])
        else:
            vols = attack_volumes_from(host.sets) if host is not None else []
        ov = vp.set_attacks(vols)
        self._attack_orphans = () if ov is None else ov.orphans
        if ov is not None:
            ov.set_visible_groups(self.visible_sets())
            ov.set_selected_group(self.selected_set)
            ov.set_selected_volume(self.selected_attack_volume
                                   if self.attacks_source == "port" else None)
        # the HOST actor beside the port draws the HOST's own sets, whatever the
        # port has authored — the right joint next to the wrong one
        vp.set_reference_attacks([] if host is None else attack_volumes_from(host.sets))
        vp.sync_attack_focus()

    def visible_sets(self):
        """Which sets the overlay shows: the selected one; else the selected
        move's; else, on the port, everything authored; on the host, nothing —
        a whole overlay's sets at once is 200 volumes of fog."""
        if self.selected_set is not None:
            return [self.selected_set]
        ps = self.pair_sets()
        if ps and self._sets_of_move_only:
            return ps
        return None if self.attacks_source == "port" else []

    def select_set(self, index: Optional[int]) -> None:
        self.selected_set = index
        self.selected_attack_volume = None
        vp = self.viewport
        if vp is not None and vp.attacks is not None:
            vp.attacks.set_visible_groups(self.visible_sets())
            vp.attacks.set_selected_group(index)
            vp.attacks.set_selected_volume(None)
            vp.sync_attack_focus()

    def select_attack_volume(self, index: Optional[int]) -> None:
        self.selected_attack_volume = index
        vp = self.viewport
        if vp is not None and vp.attacks is not None:
            vp.attacks.set_selected_volume(index if self.attacks_source == "port"
                                           else None)

    def _hitboxes_panel(self) -> None:
        """Where he hits YOU — the attack sets on the live pose, and the levers."""
        from imgui_bundle import imgui

        vp = self.viewport
        if vp is None or vp.scene is None:
            imgui.text_disabled("no scene yet")
            return
        host = self.host_attacks()
        m = self.scene.manifest

        changed = False
        for key, label in (("host", "host em%02d" % (self.host_species or 0)),
                           ("port", "this port")):
            if imgui.radio_button(label + "##atk", self.attacks_source == key):
                self.attacks_source, changed = key, True
                self.selected_attack_volume = None
            imgui.same_line()
        imgui.new_line()
        ch, self.show_attacks = imgui.checkbox("show##atk", self.show_attacks)
        imgui.same_line()
        _, vp.hitboxes_xray = imgui.checkbox("x-ray##atk", vp.hitboxes_xray)
        if ch or changed or vp.attacks is None and self.show_attacks:
            self.sync_attacks()

        if host is None:
            imgui.text_disabled("no attacks block in species/em%02d.json — build it "
                                "with tools/em_intel.py --all" % (self.host_species or 0))
            return
        _attack_provenance(imgui, self, host)
        _sets_table(imgui, self, host)
        imgui.separator()
        if self.attacks_source == "port":
            _attack_volume_editor(imgui, self, host)
            imgui.separator()
        _attack_levers(imgui, self, host)
        imgui.separator()
        _attack_actions(imgui, self, host, m)

    def _parts_panel(self) -> None:
        """Where he can be hit, and for how much — the two tables, on the live pose."""
        from imgui_bundle import imgui

        vp = self.viewport
        if vp is None or vp.scene is None:
            imgui.text_disabled("no scene yet")
            return
        host = self.host_parts()
        m = self.scene.manifest

        changed = False
        for key, label in (("host", "host em%02d" % (self.host_species or 0)),
                           ("port", "this port")):
            if imgui.radio_button(label, self.parts_source == key):
                self.parts_source, changed = key, True
            imgui.same_line()
        imgui.new_line()
        ch, self.show_parts = imgui.checkbox("show", self.show_parts)
        imgui.same_line()
        _, vp.hitboxes_xray = imgui.checkbox("x-ray", vp.hitboxes_xray)
        if ch or changed or vp.hitboxes is None and self.show_parts:
            self.sync_hitboxes()

        if self.parts_source == "host" and host is None:
            imgui.text_disabled("no species/em%02d.json — build it with "
                                "tools/em_intel.py --all"
                                % (self.host_species or 0))
            return

        _part_table(imgui, self, host)
        imgui.separator()
        if self.parts_source == "port":
            _volume_editor(imgui, self, host)
            imgui.separator()
        _grid_view(imgui, self, host)
        imgui.separator()
        _part_actions(imgui, self, host, m)

    # ---- the clip vocabulary (issue #8) ------------------------------- #
    def vocabulary(self):
        """Slot coverage, label health and the build id — computed once, on demand.

        Reads two more files (the host pack and the donor moveset) to answer "what is
        actually IN slot 37", so it is not done at construction: a session that never
        opens the Clips panel never pays for it, and a machine missing either file
        gets a partial report rather than an exception.
        """
        if self._vocab is None:
            self._vocab = _vocabulary(self.scene, getattr(self.startup, "root", None))
        return self._vocab

    # ---- editing a label (issue #8) ----------------------------------- #
    @property
    def session(self):
        """The labelling session — staging and saving live in `clips.LabelSession`.

        Kept out of this file on purpose: the write path is the part of #8 worth
        testing, and a test for it should not need a window.
        """
        if self._session is None and self.scene.manifest is not None:
            from ..clips import LabelSession
            self._session = LabelSession(self.scene.manifest, self.scene.clip_table(),
                                         self.vocabulary().build)
        return self._session

    @property
    def manifest_path(self):
        m = self.scene.manifest
        return None if m is None or m.path is None else m.path

    def _pick_clip(self, slot: int) -> None:
        """Select a slot for labelling and load its name/label into the boxes."""
        from ..clips import clip_key

        self._edit_slot = slot
        s = self.session
        entry = None if s is None else s.entry(slot)
        self._name_buf = entry.name if entry else clip_key(slot)
        self._label_buf = entry.label if entry else ""

    def stage_label(self) -> str:
        from ..manifest import ManifestError

        s = self.session
        if s is None or self._edit_slot is None:
            return "no manifest to write to"
        try:
            msg = s.stage(self._edit_slot, self._name_buf, self._label_buf)
        except ManifestError as e:
            return str(e)
        self.scene.attach_manifest(s.manifest)
        self._vocab = None                  # the label health changed
        return msg

    def save_labels(self) -> str:
        s = self.session
        if s is None:
            return "this scene has no manifest file"
        try:
            return s.save()
        except Exception as e:                                   # noqa: BLE001
            return "%s: %s" % (type(e).__name__, e)

    # ---- the action inspector (issue #9) ------------------------------ #
    @property
    def host_species(self):
        m = self.scene.manifest
        return None if m is None else m.host_species

    @property
    def browsing_species(self) -> int:
        """The overlay whose action table the panel is showing — the host by default."""
        if self._browse is None:
            self._browse = self.host_species
        return self._browse

    def browse_species(self, species: int) -> None:
        """Look at ANOTHER MHFU monster's action table.

        🔴 Browsing, not re-hosting. `(main,sub)` is dispatched by the overlay the
        ENGINE loaded for this monster, which is the one `port.host_species` selects —
        and that same number also picks the host frame PAC the porter files clips into,
        so changing it is a rebuild, not a view. The panel therefore refuses to BIND a
        pair from an overlay that is not the host, and says why.
        """
        self._browse = int(species)
        self.clear_pair()

    @property
    def intel(self):
        """`species/emNN.json` for the overlay being browsed, or None if not built."""
        species = self.browsing_species
        if species is None:
            return None
        if species not in self._intel_cache:
            from ..intel import find_intel
            self._intel_cache[species] = find_intel(species)
        return self._intel_cache[species]

    @property
    def browsing_the_host(self) -> bool:
        return self.browsing_species == self.host_species

    # ---- the host reference (issue #34) ------------------------------- #
    def host_scene(self):
        """The browsed species' own PAC, opened once. None with a reason in `status`.

        🔴 This is the HOST, not the port: `file_0<species + 6110>`. It is what the
        engine animates for the actions in the list, so it is the only thing that can
        answer "what IS (1,13)" without a cold boot.
        """
        from ..manifest import SPECIES_TO_FRAME

        species = self.browsing_species
        if species in self._host_scene:
            return self._host_scene[species]
        root = getattr(self.startup, "root", None) or "workspace"
        path = (Path(root) / "extracted" / "data_files"
                / ("file_%05d.bin" % (species + SPECIES_TO_FRAME)))
        scene = None
        if not path.exists():
            self._saved = "no host PAC at %s — docs/ASSETS.md" % path
        else:
            try:
                from ..core import open_scene
                scene = open_scene(path)
            except Exception as e:                               # noqa: BLE001
                self._saved = "%s: %s" % (type(e).__name__, e)
        self._host_scene[species] = scene
        return scene

    def host_clip_table(self):
        sc = self.host_scene()
        return {} if sc is None else sc.clip_table()

    def sync_reference(self) -> None:
        """Make the viewport's reference agree with the toggle and the browsed species.

        Called every frame from the Action panel, so flipping the checkbox or the
        species combo is all it takes — there is no second place that has to remember.
        """
        vp = self.viewport
        if vp is None:
            return
        want = self.host_scene() if self.show_host else None
        have = None if vp.reference is None else vp.reference.scene
        if want is have:
            return
        if want is None:
            vp.clear_reference(frame_camera=True)
            self._host_clip = None
            return
        vp.set_reference(want)
        self._host_clip = None
        self.follow_action()
        # the reference is attached from the ACTION panel, long after (or long
        # before) the Parts panel decided what to draw. `set_reference` replays the
        # remembered volumes; this makes the focus agree too.
        vp.sync_hitbox_focus()

    def follow_action(self) -> None:
        """Put the reference on the selected action's own clip, if one resolves.

        ⚠️ A pair can name SEVERAL a1 (`(1,13)` -> 100, 101, 117, 118) and which one
        runs depends on runtime state, so this takes the first that the host pack
        actually populates and leaves the choice visible.
        """
        vp = self.viewport
        if vp is None or vp.reference is None:
            return
        for a1 in self.host_a1():
            if a1 in self.host_clip_table():
                self.play_host_clip(a1)
                return

    def host_a1(self):
        """The executor arguments the selected pair's handler passes, in order."""
        al = self.alignment
        return list(al.pair.a1) if (al is not None and al.pair is not None) else []

    def play_host_clip(self, a1: int) -> None:
        vp = self.viewport
        if vp is None or vp.reference is None:
            return
        self._host_clip = a1
        vp.play_reference_clip(vp.reference.scene.clip(a1), 0.0)

    def host_options(self):
        """Every overlay on this machine, summarised. Read once, then cached."""
        if self._hosts is None:
            from ..intel import survey_hosts
            self._hosts = survey_hosts()
        return self._hosts

    def port_rig(self):
        """What the alignment needs to know about the rig this port ships."""
        from ..align import PortRig

        driven = set()
        for c in self.scene.clips:
            driven |= set(c.driven)
        return PortRig(n_bones=self.scene.rig.n_bones, driven=driven,
                       vertices=self._joint_vertex_counts())

    def select_pair(self, main: int, sub: int, move=None) -> None:
        self._pair, self._move = (int(main), int(sub)), move
        self._recompute_alignment()
        self.follow_action()

    def _recompute_alignment(self) -> None:
        """Re-join the host pair with whatever clip is on screen. Cheap; call freely.

        🔴 The clip's length comes from the SCENE, not from the manifest — the manifest
        may be describing a slot a rebuild has since changed, and "the handler tests
        frame 60" against a stale length is exactly the wrong answer.
        """
        from ..align import align_pair

        m = self.scene.manifest
        vp = self.viewport
        if m is None or self._pair is None:
            self.markers, self.alignment = [], None
            return
        clip = vp.clip if vp is not None else None
        entry = None
        if clip is not None and self.session is not None:
            entry = self.session.entry(clip.slot)
        self.alignment = align_pair(
            m, self._pair[0], self._pair[1], self.intel, move=self._move,
            clip=entry.name if entry else (clip.name if clip else None),
            slot=clip.slot if clip else None,
            clip_frames=clip.frames if clip else None,
            impact=entry.impact_frame if entry else None,
            allow_unentered=(m.moves[self._move].allow_unentered
                             if self._move in m.moves else False),
            rig=self.port_rig())
        self.markers = self.alignment.markers

    def set_impact_here(self) -> str:
        """Record the current frame as this clip's impact. → `clips.<n>.impact_frame`"""
        s, vp = self.session, self.viewport
        if s is None:
            return "impact frames live in the manifest — open a ports/*.toml"
        if vp is None or vp.clip is None:
            return "no clip is playing"
        from ..manifest import ManifestError
        slot = vp.clip.slot
        entry = s.entry(slot)
        frame = int(round(vp.playback.phase))
        try:
            msg = s.stage(slot, entry.name if entry else self._name_buf,
                          entry.label if entry else self._label_buf,
                          impact_frame=frame)
        except ManifestError as e:
            return str(e)
        self.scene.attach_manifest(s.manifest)
        self._vocab = None
        self._recompute_alignment()
        return "impact = frame %d.  %s" % (frame, msg)

    def _action_panel(self) -> None:
        """What the HOST action expects, against the clip on screen. → issue #9"""
        from imgui_bundle import imgui

        m = self.scene.manifest
        if m is None:
            imgui.text_wrapped("An action is a binding between a HOST behaviour pair "
                               "and one of this port's clips, so it needs a manifest. "
                               "Open a ports/*.toml.")
            return
        self._species_row(imgui, m)
        self.sync_reference()
        if self.intel is None:
            imgui.text_wrapped("no species/em%02d.json — build the action intel with:"
                               % self.browsing_species)
            imgui.text_disabled("  python tools/em_intel.py --all")
            return

        self._moves_row(imgui, m)
        if self.alignment is not None:
            _alignment_view(imgui, self.alignment, self)
            self._bind_row(imgui)
            imgui.separator()
        self._pair_table(imgui)

    def _moves_panel(self) -> None:
        """The pairs as a graph: what the engine walks after each one. → ui/graph.py

        A tab beside the Viewport rather than a strip under the Action panel, because
        a chain with its guards needs the width — and because the graph and the model
        are looked at in turn, not side by side: pick the pair here, watch it there.
        """
        from imgui_bundle import imgui

        m = self.scene.manifest
        if m is None:
            imgui.text_wrapped("The graph is the HOST overlay's — which host needs a "
                               "manifest. Open a ports/*.toml.")
            return
        if self.intel is None:
            imgui.text_wrapped("no species/em%02d.json — build the action intel with:"
                               % self.browsing_species)
            imgui.text_disabled("  python tools/em_intel.py --all")
            return
        self._graph.draw(imgui, self)

    def _species_row(self, imgui, m) -> None:
        """Which overlay's actions you are looking at, and the 17 you could ride.

        🔴 A ported monster has no AI of its own — MHP3rd ships a model, a skeleton and
        animations, and nothing else. The engine loads ONE MHFU species overlay for it,
        `port.host_species` picks which, and every `(main,sub)` in this panel is that
        host's. Riding the Tigrex is a choice among 17, not a fact about the Zinogre.
        """
        from ..intel import available

        ids = available()
        if not ids:
            return
        cur = self.browsing_species
        idx = ids.index(cur) if cur in ids else 0
        imgui.set_next_item_width(110.0)
        changed, pick = imgui.combo("##species", idx,
                                    ["em%02d%s" % (s, "  (host)" if s == self.host_species
                                                   else "") for s in ids])
        if changed and ids[pick] != cur:
            self.browse_species(ids[pick])
        imgui.same_line()
        if self.browsing_the_host:
            imgui.text_disabled("the host this port rides (port.host_species = %d)"
                                % self.host_species)
        else:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.70, 0.20, 1.0))
            imgui.text_wrapped(
                "! browsing em%02d — this port rides em%02d, so these pairs are NOT "
                "the ones its engine dispatches. Comparing hosts, not binding: "
                "host_species also picks the host frame PAC the porter files clips "
                "into, so changing it is a rebuild."
                % (cur, self.host_species))
            imgui.pop_style_color()
        changed, self.show_host = imgui.checkbox("show em%02d beside the port"
                                                 % cur, self.show_host)
        if changed and not self.show_host:
            self.sync_reference()
        imgui.same_line()
        imgui.text_disabled("what the action actually looks like")
        if imgui.is_item_hovered():
            imgui.set_tooltip("Loads the host species' own PAC and plays the clip its "
                              "handler passes to the executor, beside your port. Both "
                              "run at the same rate; each loops at its OWN end.")
        self._hosts_table(imgui, ids)

    def _hosts_table(self, imgui, ids) -> None:
        """The 17 overlays side by side — the "which host should this port ride?" view."""
        if not imgui.collapsing_header("compare the %d MHFU action tables" % len(ids)):
            return
        imgui.text_wrapped(
            "Every overlay has the identical action tick: 8 mains, each fanning into a "
            "sub_state switch. What differs is how much vocabulary you inherit. "
            "'timed' pairs test fixed clip frames your animation has to hit; 'budget' "
            "ones end on the +0x414 countdown instead of on your clip.")
        flags = (imgui.TableFlags_.borders_inner_h.value
                 | imgui.TableFlags_.row_bg.value
                 | imgui.TableFlags_.sizing_stretch_prop.value)
        if not imgui.begin_table("##hosts", 6, flags):
            return
        for name in ("overlay", "pairs", "timed", "budget", "fx", "opaque"):
            imgui.table_setup_column(name)
        imgui.table_headers_row()
        for h in self.host_options():
            imgui.table_next_row()
            imgui.table_next_column()
            if imgui.selectable("em%02d%s##h%d" % (h.species,
                                                   " *" if h.species == self.host_species
                                                   else "", h.species),
                                h.species == self.browsing_species,
                                imgui.SelectableFlags_.span_all_columns.value)[0]:
                self.browse_species(h.species)
            for value in (h.pairs, h.timed, h.budget, h.effects):
                imgui.table_next_column()
                imgui.text(str(value))
            imgui.table_next_column()
            imgui.text_disabled("%d/8" % h.opaque_mains)
            if imgui.is_item_hovered():
                imgui.set_tooltip("mains with no visible sub_state jump table — "
                                  "'we cannot see it', which is not 'it does not "
                                  "exist'")
        imgui.end_table()

    def clear_pair(self) -> None:
        """Back to the list of pairs, keeping the clip and the camera where they are."""
        self._pair, self._move, self.alignment, self.markers = None, None, None, []

    def _bind_row(self, imgui) -> None:
        """Write the alignment down: `[moves.<name>] main/sub/clip`.

        Refused outright for a pair the census MEASURED as never entered — the same
        thing `validate` refuses, and for the same reason. `allow_unentered = true` is
        the override, and it belongs in the file next to a comment saying why, not
        behind a button here.
        """
        al, s = self.alignment, self.session
        if al is None or s is None:
            return
        if not self.browsing_the_host:
            imgui.text_disabled("cannot bind: this pair belongs to em%02d, and the "
                                "engine dispatches em%02d for this port"
                                % (self.browsing_species, self.host_species))
            return
        blocked = [f for f in al.errors if f.code == "NEVER_ENTERED"]
        imgui.set_next_item_width(140.0)
        _, self._bind_buf = imgui.input_text("##bindname", self._bind_buf)
        imgui.same_line()
        if blocked:
            imgui.text_disabled("cannot bind: the census says the engine never "
                                "enters this pair")
            return
        if imgui.small_button("bind as move"):
            from ..manifest import ManifestError
            try:
                self._saved = s.stage_move(self._bind_buf or "move_%d_%d" % (al.main,
                                                                             al.sub),
                                           al.main, al.sub, al.clip)
                self.scene.attach_manifest(s.manifest)
                self.select_pair(al.main, al.sub, self._bind_buf or None)
            except ManifestError as e:
                self._saved = str(e)
        imgui.same_line()
        imgui.text_disabled("-> [moves] on (%d,%d)%s" % (al.main, al.sub,
                                                         " / %s" % al.clip
                                                         if al.clip else ""))

    def _moves_row(self, imgui, m) -> None:
        if not m.moves:
            imgui.text_wrapped("[moves] is empty — pick a pair below to try it "
                               "against the clip on screen; bind it when it fits.")
            return
        imgui.text_disabled("moves")
        for name in sorted(m.moves):
            mv = m.moves[name]
            imgui.same_line()
            if imgui.small_button("%s (%d,%d)" % (name, mv.main, mv.sub)):
                if mv.clip and self.viewport is not None:
                    try:
                        self.viewport.play_clip(self.scene.clip(mv.clip))
                        self._pick_clip(self.scene.clip(mv.clip).slot)
                    except KeyError:
                        pass
                self.select_pair(mv.main, mv.sub, name)

    #: the pair table never gets less than this many points of height. Below about
    #: this it is a header and one row, which reads as "the list is gone".
    _PAIR_TABLE_MIN = 150.0

    def _pair_table(self, imgui) -> None:
        """Every `(main,sub)` the overlay dispatches — the deciding view.

        Choosing the pair is the authoring act and it happens BEFORE a move exists:
        `ports/zinogre.toml` ships with `[moves]` deliberately empty because aligning
        an unlabelled vocabulary is guessing. So the table is browsable and the
        alignment updates against whatever clip is playing.
        """
        imgui.set_next_item_width(120.0)
        _, self._pair_filter = imgui.input_text("##pf", self._pair_filter)
        imgui.same_line()
        imgui.text_disabled("filter %d pairs em%02d dispatches"
                            % (len(self.intel), self.browsing_species))

        flags = (imgui.TableFlags_.borders_inner_h.value
                 | imgui.TableFlags_.row_bg.value
                 | imgui.TableFlags_.scroll_y.value
                 | imgui.TableFlags_.sizing_stretch_prop.value)
        # 🔴 An explicit height, not "whatever is left". `begin_table` with ScrollY and
        # a zero outer size takes the REMAINDER of the panel, and after a headline and
        # a dozen findings that remainder is a header and one row — which reads as
        # "selecting a pair made the list of pairs disappear". With a floor the panel
        # scrolls to it instead.
        height = max(self._PAIR_TABLE_MIN, imgui.get_content_region_avail().y)
        if not imgui.begin_table("##pairs", 5, flags, imgui.ImVec2(0.0, height)):
            return
        for name, w in (("pair", 0.8), ("ends on", 0.9), ("tests", 1.0), ("fx", 0.3),
                        ("after", 1.1)):
            imgui.table_setup_column(name, imgui.TableColumnFlags_.width_stretch.value, w)
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_headers_row()
        needle = self._pair_filter.strip().lower()
        for p in self.intel:
            gates = p.tested_frames
            nxt = " ".join("(%d,%d)" % t for t in p.successors[:4])
            row = "%d,%d %s %s %s" % (p.main, p.sub, p.ends_on,
                                      " ".join("%g" % f for f in gates), nxt)
            if needle and needle not in row.lower():
                continue
            imgui.table_next_row()
            imgui.table_next_column()
            sel = self._pair == (p.main, p.sub)
            if imgui.selectable("(%d,%d)##p%d_%d" % (p.main, p.sub, p.main, p.sub),
                                sel, imgui.SelectableFlags_.span_all_columns.value)[0]:
                self.select_pair(p.main, p.sub)
            imgui.table_next_column()
            if p.budget.gated:
                imgui.push_style_color(imgui.Col_.text,
                                       imgui.ImVec4(0.98, 0.70, 0.20, 1.0))
                imgui.text("budget")
                imgui.pop_style_color()
            else:
                imgui.text_disabled(p.ends_on or "?")
            imgui.table_next_column()
            imgui.text(", ".join("%g" % f for f in gates[:4]) or "·")
            imgui.table_next_column()
            imgui.text(str(len(p.effects)) if p.effects else "")
            imgui.table_next_column()
            # where the handler sends him when the action ends (static). Blank on a
            # handled pair = it never ends itself; "?" = the file predates the join.
            if p.next is None:
                imgui.text_disabled("?")
            elif not p.next:
                imgui.text_disabled("holds")
            else:
                imgui.text_disabled(nxt + (" +" if len(p.successors) > 4 else ""))
            if imgui.is_item_hovered() and p.next:
                imgui.set_tooltip("\n".join(str(e) for e in p.next))
        imgui.end_table()

    # ---- clips (issues #7, #8) ---------------------------------------- #
    def _clips_panel(self) -> None:
        """Every clip in the PAC: what is in the slot, and what we call it.

        The `cov` column is the one that is not obvious from the file. A **FILLER**
        slot holds a copy of the idle clip — 30 of the built Zinogre's 64 do — and
        forcing that a1 plays idle, which on screen is identical to the override
        never firing. Every "the latch didn't work" report has to rule that out
        first, and this is where it gets ruled out.
        """
        from imgui_bundle import imgui

        vp = self.viewport
        if vp is None:
            imgui.text_disabled("no GL context yet")
            return
        clips = self.scene.clips
        if not clips:
            imgui.text_disabled("this PAC has no animation sub-resource")
            return
        vocab = self.vocabulary()

        imgui.text("%d clips" % len(clips))
        imgui.same_line()
        imgui.text_disabled("%d looping" % sum(1 for c in clips if c.loop))
        if vocab.build:
            imgui.text_disabled(vocab.build)
            if imgui.is_item_hovered():
                imgui.set_tooltip("the build a label typed here is keyed to. Clip ids "
                                  "are PER BUILD: rebuild and they shift.")
        _coverage_line(imgui, vocab)
        for n in vocab.notes:
            imgui.text_disabled("• %s" % n)
        _label_health(imgui, vocab)

        _, self._clip_filter = imgui.input_text("##filter", self._clip_filter)
        imgui.same_line()
        imgui.text_disabled("filter")

        flags = (imgui.TableFlags_.borders_inner_h.value
                 | imgui.TableFlags_.row_bg.value
                 | imgui.TableFlags_.scroll_y.value
                 | imgui.TableFlags_.sizing_stretch_prop.value)
        # leave room for the label editor under the table; it is the point of #8 and
        # must not be the thing that scrolls off the bottom.
        editor_h = 150.0 if self.scene.manifest is not None else 34.0
        if not imgui.begin_table("##clips", 6, flags,
                                 imgui.ImVec2(0.0, -editor_h)):
            return
        # `a1` gets its own column even when the clip is named: the executor argument
        # IS the slot index, so it is the number a mod script types, and a renamed
        # clip ("charge") no longer carries it in its name.
        for name, w in (("a1", 0.45), ("name", 1.0), ("cov", 0.7), ("frames", 0.75),
                        ("loop", 0.45), ("travel", 0.6)):
            imgui.table_setup_column(name, imgui.TableColumnFlags_.width_stretch.value, w)
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_headers_row()

        needle = self._clip_filter.strip().lower()
        current = vp.clip.slot if vp.clip is not None else None
        for c in clips:
            cov = vocab.coverage.slots.get(c.slot)
            label = "%d %s %s" % (c.slot, " ".join(c.names),
                                  (cov.kind if cov else ""))
            entry = self._manifest_clip(c.slot)
            if entry is not None:
                label += " " + entry.label
            if needle and needle not in label.lower():
                continue
            imgui.table_next_row()
            imgui.table_next_column()
            if imgui.selectable(
                    "%d##c%d" % (c.slot, c.slot), current == c.slot,
                    imgui.SelectableFlags_.span_all_columns.value)[0]:
                vp.play_clip(c)
                self._pick_clip(c.slot)
            if imgui.is_item_hovered():
                imgui.set_tooltip("%s\n%.2f s at speed %.2f%s"
                                  % (cov.why() if cov else "no coverage verdict",
                                     wall_clock(c.frames, vp.playback.speed),
                                     vp.playback.speed,
                                     "\n" + entry.label if entry and entry.label
                                     else ""))
            imgui.table_next_column()
            if c.names:
                imgui.text(c.name)
            else:
                imgui.text_disabled("—")
            imgui.table_next_column()
            _coverage_cell(imgui, cov)
            imgui.table_next_column()
            imgui.text(str(c.frames))
            imgui.table_next_column()
            imgui.text("loop" if c.loop else "")
            imgui.table_next_column()
            net, peak = self._travel(c)
            imgui.text("%.0f" % net if net >= 1.0 else "·")
            if not c.whole_rig and imgui.is_item_hovered():
                imgui.set_tooltip("partial: present in only some joint-partition streams")
        imgui.end_table()
        imgui.separator()
        self._label_editor()

    def _manifest_clip(self, slot: int):
        m = self.scene.manifest
        if m is None:
            return None
        return next((c for c in m.clips.values() if c.slot == slot), None)

    def _label_editor(self) -> None:
        """Name a clip, and put the name in the manifest keyed to THIS build. → #8"""
        from imgui_bundle import imgui

        m = self.scene.manifest
        if m is None:
            imgui.text_disabled("labels need a manifest — open a ports/*.toml")
            return
        if self._edit_slot is None:
            imgui.text_disabled("pick a clip to name it")
            return
        slot = self._edit_slot
        cov = self.vocabulary().coverage.slots.get(slot)
        imgui.text("slot %d" % slot)
        imgui.same_line()
        _coverage_cell(imgui, cov)
        if cov is not None and imgui.is_item_hovered():
            imgui.set_tooltip(cov.why())

        imgui.set_next_item_width(-64.0)
        _, self._name_buf = imgui.input_text("name", self._name_buf)
        imgui.set_next_item_width(-64.0)
        _, self._label_buf = imgui.input_text("label", self._label_buf)

        if imgui.button("stage"):
            self._saved = self.stage_label()
        imgui.same_line()
        pending = self.session.pending if self.session else 0
        if pending:
            if imgui.button("save %d to %s" % (pending, self.manifest_path.name)):
                self._saved = self.save_labels()
            imgui.same_line()
            if imgui.button("discard"):
                self.session.discard()
                self._saved = "discarded the pending edits (the file was not touched)"
        else:
            imgui.text_disabled("nothing pending")
        if self._saved:
            imgui.text_wrapped(self._saved)

    def _travel(self, clip):
        """`root_travel`, cached — it evaluates the clip, so not once per frame."""
        got = self._travel_cache.get(clip.slot)
        if got is None:
            got = root_travel(self.scene, clip)
            self._travel_cache[clip.slot] = got
        return got

    def _timeline_panel(self) -> None:
        """Transport, scrubber and the rate — the engine's model, not a media player."""
        from imgui_bundle import imgui

        vp = self.viewport
        if vp is None or vp.clip is None:
            imgui.text_disabled("no clip — pick one in Clips")
            return
        pb = vp.playback

        if imgui.button("|<"):
            pb.rewind()
            vp.set_pose(vp.clip, pb.phase)
        imgui.same_line()
        if imgui.button("<|"):
            pb.step(-1)
            vp.set_pose(vp.clip, pb.phase)
        imgui.same_line()
        if imgui.button("pause" if pb.playing else "play "):
            pb.toggle()
        imgui.same_line()
        if imgui.button("|>"):
            pb.step(1)
            vp.set_pose(vp.clip, pb.phase)
        imgui.same_line()
        _, pb.loop = imgui.checkbox("loop", pb.loop)
        imgui.same_line()
        _, strip = imgui.checkbox("in place", vp.strip_root)
        if strip != vp.strip_root:
            vp.strip_root = strip
            vp.set_pose(vp.clip, pb.phase)

        moved, frame = imgui.slider_float("##scrub", pb.phase, 0.0, max(pb.end, 1.0),
                                          "frame %.1f / " + str(int(pb.end)))
        if moved:
            pb.seek(frame)
            vp.set_pose(vp.clip, pb.phase)

        # 🔴 the rate is the ACTION's, not the clip's: 2.0 and 2.4 were both measured
        # on one monster. The clip owns only the frame SPAN.
        changed, speed = imgui.slider_float("speed", pb.speed, 0.25, 4.0, "%.2f f/frame")
        if changed:
            pb.speed = max(0.05, speed)
        for s in OBSERVED_SPEEDS:
            imgui.same_line()
            if imgui.small_button("%.1f" % s):
                pb.speed = s
        imgui.text_disabled(
            "%.2f s at %.2f  (end %d / speed / %g Hz) — the SPAN is the clip's, the "
            "RATE comes from the action dispatch" % (pb.duration, pb.speed, pb.end,
                                                     GAME_HZ))
        net, peak = self._travel(self.scene.clip(vp.clip))
        imgui.text_disabled("root travel  net %.0f  peak %.0f%s"
                            % (net, peak, "" if net >= 1.0 else "   (in place)"))

        # 🔴 the alignment answer needs ONE number the file cannot give: where this
        # clip's own contact is. Scrub to it and press this. → clips.<n>.impact_frame
        if self.scene.manifest is not None:
            entry = self.session.entry(vp.clip.slot) if self.session else None
            if imgui.small_button("set impact = frame %d" % int(round(pb.phase))):
                self._saved = self.set_impact_here()
            imgui.same_line()
            if entry is not None and entry.impact_frame is not None:
                imgui.text_disabled("impact %d" % entry.impact_frame)
            else:
                imgui.text_disabled("no impact frame recorded for this clip")
        self._marker_strip(imgui, pb)

    def _marker_strip(self, imgui, pb) -> None:
        """The host action's frames, drawn in the CLIP's own frame space. → #9

        🔴 A gate past the clip's last frame is drawn HOLLOW at the right edge rather
        than clamped silently: the cursor never reaches it, so that branch of the
        handler never runs, and a tick indistinguishable from a reachable one would
        hide the single most expensive thing this panel can tell you.
        """
        from ..align import EFFECT, GATE, IMPACT, OURS, WINDOW

        h = 22.0
        w = max(imgui.get_content_region_avail().x, 1.0)
        pos = imgui.get_cursor_screen_pos()
        draw = imgui.get_window_draw_list()
        bg = imgui.get_color_u32(imgui.ImVec4(0.18, 0.19, 0.22, 1.0))
        draw.add_rect_filled(pos, imgui.ImVec2(pos.x + w, pos.y + h), bg, 3.0)
        hovered, mouse = None, imgui.get_io().mouse_pos
        if pb.end > 0:
            def at(mk):
                return pos.x + w * max(0.0, min(mk.frame / pb.end, 1.0))

            for mk in self.markers:
                x = at(mk)
                col = imgui.get_color_u32(imgui.ImVec4(*_MARKER_COLOR[mk.kind]))
                if mk.unreachable:
                    # ⚠️ KEYWORDS, not positional. ImGui's C++ AddRect takes
                    # (rounding, flags, thickness) and imgui_bundle's binding takes
                    # (rounding, thickness, flags) — passing the C++ order hands a
                    # float to `flags: int` and raises at the first UNREACHABLE
                    # marker drawn, which is the moment an impact frame is set.
                    draw.add_rect(imgui.ImVec2(x - 2, pos.y + 3),
                                  imgui.ImVec2(x + 2, pos.y + h - 3), col,
                                  thickness=1.5)
                else:
                    draw.add_line(imgui.ImVec2(x, pos.y + 1),
                                  imgui.ImVec2(x, pos.y + h - 1), col, 2.0)
                if abs(mouse.x - x) < 6 and pos.y <= mouse.y <= pos.y + h:
                    hovered = mk

            # ⚠️ The labels have to be placed, not just drawn. On a 408-frame clip the
            # Tigrex's 40/50/58/60 gates land inside 5% of the width and their texts
            # overprint into an unreadable smear. The IMPACT label is placed first
            # because it is the answer the panel exists to give; the rest take the room
            # that is left, and the tick + hover tooltip carry the ones that are
            # dropped.
            taken = []
            for mk in sorted(self.markers,
                             key=lambda k: (k.kind != IMPACT, k.frame)):
                x = at(mk) + 3.0
                tw = imgui.calc_text_size(mk.label).x
                if any(x < b and x + tw > a for a, b in taken):
                    continue
                taken.append((x - 2.0, x + tw + 2.0))
                draw.add_text(imgui.ImVec2(x, pos.y + 3),
                              imgui.get_color_u32(
                                  imgui.ImVec4(*_MARKER_COLOR[mk.kind])), mk.label)
            x = pos.x + w * pb.progress
            head = imgui.get_color_u32(imgui.ImVec4(0.95, 0.25, 0.28, 1.0))
            draw.add_line(imgui.ImVec2(x, pos.y), imgui.ImVec2(x, pos.y + h), head, 2.0)
        imgui.dummy(imgui.ImVec2(w, h))
        if hovered is not None:
            imgui.set_tooltip(plain(
                "frame %g — %s%s"
                % (hovered.frame, hovered.detail or hovered.label,
                   "\n! past the clip's last frame: never reached"
                   if hovered.unreachable else "")))
        if not self.markers:
            imgui.text_disabled("no frame markers — pick a (main,sub) in Action and "
                                "the handler's own frames appear here")
            return
        first = True
        for kind, name in ((GATE, "cursor >= F"), (WINDOW, "window edge"),
                           (EFFECT, "host effect"), (OURS, "our effect"),
                           (IMPACT, "impact")):
            if not any(mk.kind == kind for mk in self.markers):
                continue
            if not first:
                imgui.same_line()
            first = False
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_MARKER_COLOR[kind]))
            imgui.text("| " + name)
            imgui.pop_style_color()

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
            # a shown volume under the cursor wins over a joint: with the gizmos
            # on, the click is "this sphere", and the joint is still one key away
            picked = None
            picked_attack = None
            if self.show_attacks and self.attacks_source == "port" \
                    and vp.attacks is not None:
                v = vp.attacks.pick(mvp, size, m.x - pos.x, m.y - pos.y)
                picked_attack = None if v is None else vp.attacks.index_of(v)
            if picked_attack is None and self.show_parts \
                    and self.parts_source == "port" and vp.hitboxes is not None:
                v = vp.hitboxes.pick(mvp, size, m.x - pos.x, m.y - pos.y)
                picked = None if v is None else vp.hitboxes.index_of(v)
            if picked_attack is not None:
                self.select_attack_volume(
                    None if self.selected_attack_volume == picked_attack
                    else picked_attack)
            elif picked is not None:
                self.select_volume(None if self.selected_volume == picked else picked)
            else:
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
        if self.ini_folder is None:
            p.ini_folder_type = hello_imgui.IniFolderType.app_user_config_folder
            p.ini_filename = "%s.ini" % INI_NAME
        else:
            p.ini_folder_type = hello_imgui.IniFolderType.absolute_path
            p.ini_filename = "%s/%s.ini" % (self.ini_folder, INI_NAME)

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

    def split(initial, new, direction, ratio):
        sp = hello_imgui.DockingSplit()
        sp.initial_dock = initial
        sp.new_dock = new
        sp.direction = direction
        sp.ratio = ratio
        return sp

    # the viewport keeps the middle; inspectors left, clips right, timeline under.
    splits = [split("MainDockSpace", "Left", imgui.Dir.left, 0.18),
              split("MainDockSpace", "Right", imgui.Dir.right, 0.20),
              split("MainDockSpace", "Bottom", imgui.Dir.down, 0.30),
              # the action inspector goes BESIDE the timeline, not in the narrow right
              # column: its findings are prose, and it has to be readable at the same
              # time as the marker strip it fills — tabbing it with either would hide
              # one half of the comparison the panel exists to make.
              split("Bottom", "BottomRight", imgui.Dir.right, 0.45)]

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
    # the move graph shares the viewport's dock space: a TAB beside it, so the two
    # views swap with a click and neither is squeezed into an inspector column
    graph = win("Moves", "MainDockSpace", app._moves_panel)
    graph.imgui_window_flags = _no_scroll_flags()

    d = hello_imgui.DockingParams()
    d.docking_splits = splits
    d.dockable_windows = [viewport, graph,
                          win("Timeline", "Bottom", app._timeline_panel, focus=True),
                          win("Scene", "Left", app._scene_panel),
                          win("View", "Left", app._view_panel),
                          win("Joints", "Left", app._joints_panel),
                          win("Clips", "Right", app._clips_panel),
                          win("Parts", "Right", app._parts_panel),
                          win("Hitboxes", "Right", app._hitboxes_panel),
                          win("Action", "BottomRight", app._action_panel)]
    return d


def _no_scroll_flags() -> int:
    from imgui_bundle import imgui
    return (imgui.WindowFlags_.no_scrollbar.value
            | imgui.WindowFlags_.no_scroll_with_mouse.value)


def _status(app: EditorApp) -> None:
    from imgui_bundle import imgui
    imgui.text(app.status or "attaching to the GL context…")


#: hello_imgui's default font covers Latin-1 and general punctuation and NOTHING else,
#: so a dingbat or an emoji draws as a replacement box. The finding messages are shared
#: with the CLI, where the symbols carry real weight, so they are translated on the way
#: to the SCREEN rather than removed at the source.
_GLYPHS = {"\U0001f534": "[!]", "\u26a0\ufe0f": "[!]", "\u26a0": "[!]",
           "\u2716": "x", "\u2714": "ok", "\u25cf": "*", "\u2192": "->"}


def plain(text: str) -> str:
    """Text the default imgui font can actually draw, at a length a panel can hold.

    Also shortens absolute paths: `em_intel` records *why* a census is absent as the
    full path it looked at, which is right in a log and is half a panel on screen.
    """
    import os

    for bad, good in _GLYPHS.items():
        text = text.replace(bad, good)
    for root, short in ((os.getcwd() + os.sep, ""),
                        (os.path.expanduser("~") + os.sep, "~" + os.sep)):
        if root not in (os.sep, ""):
            text = text.replace(root, short)
    return text


def _framebuffer_scale() -> float:
    """Framebuffer pixels per imgui point, as the backend reports it."""
    from imgui_bundle import imgui
    fb = imgui.get_io().display_framebuffer_scale
    return float(max(fb.x, 1.0))


# --------------------------------------------------------------------------- #
# the clip vocabulary (issue #8)
# --------------------------------------------------------------------------- #
#: how a slot's coverage verdict is painted. FILLER is the one that has to be loud:
#: it is a successful override onto the idle clip, and it looks exactly like failure.
_COV_COLOR = {"CARRIED": (0.55, 0.82, 0.55, 1.0),
              "FILLER": (0.98, 0.70, 0.20, 1.0),
              "HOST": (0.60, 0.72, 0.98, 1.0),
              "ALTERED": (0.90, 0.55, 0.95, 1.0),
              "UNKNOWN": (0.55, 0.58, 0.64, 1.0)}


def _vocabulary(scene, root: Optional[str] = None):
    """Read the two companion packs and classify the open one's slots.

    A missing companion is a NOTE, never an exception: `workspace/` is a machine-local
    symlink into a game dump, so a checkout without the donor moveset must still open
    the editor — it just cannot tell a carried clip from the porter's filler, and says
    so instead of guessing.
    """
    from ..clips import build_id, clip_table, source_clip_table, survey

    m, notes = scene.manifest, []
    build = None
    if scene.path is not None:
        try:
            build = build_id(scene.path)
        except OSError as e:                                     # pragma: no cover
            notes.append("no build id: %s" % e)
    host = source = None
    if m is not None and scene.game == "mhfu":
        for what, path, reader in (
                ("host pack", m.host_pac_path(root or "workspace"), clip_table),
                ("donor moveset", m.source_paths(root or "workspace")["anim"],
                 source_clip_table)):
            if not path.exists():
                notes.append("no %s at %s — docs/ASSETS.md" % (what, path))
                continue
            try:
                got = reader(path.read_bytes())
            except Exception as e:                               # noqa: BLE001
                notes.append("%s unreadable (%s)" % (what, e))
                continue
            if reader is clip_table:
                host = got
            else:
                source = got
    elif m is None:
        notes.append("no manifest: slots cannot be classified, and a label typed here "
                     "would have nowhere to go")
    elif scene.game != "mhfu":
        notes.append("this is the DONOR pack — coverage describes a built port")
    return survey(m, scene.clip_table(), host, source, build, notes)


def _is_bare_key(name: str) -> bool:
    """A TOML bare key, which is what `[clips.<name>]` needs the typed name to be."""
    return bool(name) and all(ch.isalnum() or ch in "-_" for ch in name)


def _coverage_line(imgui, vocab) -> None:
    n = vocab.coverage.counts()
    if not vocab.coverage.has_source:
        return
    bits = [("%d carried" % n["CARRIED"], "CARRIED"),
            ("%d filler" % n["FILLER"], "FILLER"),
            ("%d host" % n["HOST"], "HOST"),
            ("%d altered" % n["ALTERED"], "ALTERED")]
    for i, (text, kind) in enumerate(bits):
        if i:
            imgui.same_line()
        if not n[kind]:
            imgui.text_disabled(text)
            continue
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_COV_COLOR[kind]))
        imgui.text(text)
        imgui.pop_style_color()
    if vocab.coverage.dropped:
        imgui.text_disabled("%d donor clip(s) dropped: %s"
                            % (len(vocab.coverage.dropped),
                               ", ".join(str(s) for s
                                         in sorted(vocab.coverage.dropped))))
        if imgui.is_item_hovered():
            imgui.set_tooltip("the host pack has no slot of that index, so the "
                              "porter had nowhere to file them. They are not in "
                              "this build at all.")


def _coverage_cell(imgui, cov) -> None:
    if cov is None:
        imgui.text_disabled("-")
        return
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_COV_COLOR[cov.kind]))
    imgui.text(cov.kind[:7].lower())
    imgui.pop_style_color()
    if imgui.is_item_hovered():
        imgui.set_tooltip(cov.why())


def _label_health(imgui, vocab) -> None:
    """Only speaks up when a label has stopped meaning what it says."""
    bad = vocab.suspect
    if not bad:
        return
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.70, 0.20, 1.0))
    imgui.text_wrapped("! %d label(s) do not match this build" % len(bad))
    imgui.pop_style_color()
    for t in bad:
        imgui.bullet_text("%s (slot %d): %s" % (t.name, t.slot, t.status))
        if imgui.is_item_hovered():
            imgui.set_tooltip(t.message)


# --------------------------------------------------------------------------- #
# the action inspector (issue #9)
# --------------------------------------------------------------------------- #
_MARKER_COLOR = {"gate": (0.98, 0.70, 0.20, 0.95),
                 "window": (0.55, 0.82, 0.98, 0.95),
                 "effect": (0.85, 0.55, 0.98, 0.95),
                 "ours": (0.55, 0.90, 0.60, 0.95),
                 "impact": (0.98, 0.35, 0.38, 1.0)}

_LEVEL_COLOR = {"error": (0.98, 0.42, 0.42, 1.0),
                "warn": (0.98, 0.75, 0.30, 1.0),
                "info": (0.62, 0.68, 0.78, 1.0)}


def _alignment_view(imgui, al, app) -> None:
    """The headline, then every finding. The headline IS the feature."""
    if imgui.small_button("< all pairs"):
        app.clear_pair()
        return
    imgui.same_line()
    imgui.text("%s  ->  (%d,%d)" % (al.move, al.main, al.sub))
    if al.clip:
        imgui.same_line()
        imgui.text_disabled("clip %s%s" % (al.clip,
                                           "  %df" % al.frames if al.frames else ""))
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.92, 0.94, 0.98, 1.0))
    imgui.text_wrapped(plain(al.headline))
    imgui.pop_style_color()
    _host_clip_row(imgui, al, app)
    _hits_with_row(imgui, al, app)
    _then_row(imgui, al, app)

    # The findings are collapsed unless something is actually WRONG. They are prose,
    # a dozen of them is normal, and left open they push the pair table off the panel —
    # while the one line anybody reads is the headline above.
    if al.findings:
        n = len(al.errors), len(al.warnings)
        label = "%d finding%s%s###findings" % (
            len(al.findings), "" if len(al.findings) == 1 else "s",
            "  —  %d error, %d warning" % n if any(n) else "")
        imgui.set_next_item_open(bool(al.errors), imgui.Cond_.once.value)
        open_ = imgui.collapsing_header(label)
        if not open_:
            _finding_dots(imgui, al)          # the summary stands in for the list
        if open_:
            for f in al.findings:
                imgui.push_style_color(imgui.Col_.text,
                                       imgui.ImVec4(*_LEVEL_COLOR[f.level]))
                imgui.text_wrapped("%s  %s" % ("x" if f.level == "error"
                                               else "!" if f.level == "warn" else "-",
                                               plain(f.message)))
                imgui.pop_style_color()
            if al.pair is not None and al.pair.handler:
                imgui.text_disabled("handler 0x%08X   a1 %s" % (
                    al.pair.handler, ",".join(str(x) for x in al.pair.a1) or "-"))
            _species_effects(imgui, app)


def _then_row(imgui, al, app) -> None:
    """What the ENGINE does after this pair, beside what the move DECLARES.

    The two are different facts and both are shown: the handler's own hand-off is
    static intel (`PairIntel.next`), the manifest's `after =` is the modder's choice.
    A move on a pair that ends itself needs no `after`; a move on a pair that does
    NOT (`holds`) is what parks with its hitbox spent, and says so here.
    """
    if al.pair is None or al.pair.next is None:
        return
    p = al.pair
    if not p.next:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.70, 0.20, 1.0))
        imgui.text_wrapped("! (%d,%d) never ends by itself: forced, it stays until "
                           "something else moves him. A move here needs `after =`."
                           % (p.main, p.sub))
        imgui.pop_style_color()
    else:
        imgui.text_disabled("engine: after")
        for e in p.next:
            imgui.same_line()
            tgt = "/".join("(%d,%d)" % t for t in e.to) or "?"
            if imgui.small_button("%s##then%s" % (tgt, e.site)):
                if e.to:
                    app.select_pair(*e.to[0])
                    return
            if imgui.is_item_hovered():
                imgui.set_tooltip(str(e))
            if e.reason:
                imgui.same_line()
                imgui.text_disabled(e.reason)
    m = app.scene.manifest
    mv = m.moves.get(al.move) if (m is not None and al.move) else None
    if mv is not None and getattr(mv, "after", None):
        nxt = m.moves.get(mv.after)
        imgui.text_disabled("declared after = %s%s" % (
            mv.after, " (%d,%d)" % (nxt.main, nxt.sub) if nxt else "  (no such move!)"))
    imgui.same_line()
    if imgui.small_button("open in Moves tab"):
        from imgui_bundle import hello_imgui
        w = hello_imgui.get_runner_params().docking_params.dockable_window_of_name("Moves")
        if w is not None:
            w.focus_window_at_next_frame = True


def _host_clip_row(imgui, al, app) -> None:
    """Which clip the HOST plays for this action — the answer to "what IS (1,13)".

    🔴 Only 110 of em75's 225 pairs that name an a1 name one the Tigrex's own pack
    populates; the rest name slots 84+ that are simply not there (issue #34 has the
    numbers). A missing one is drawn disabled and SAYS the slot is absent, because
    silently falling through to another a1 would show the wrong animation for the
    action — which is the one mistake this whole panel exists to prevent.
    """
    if al.pair is None or not al.pair.a1:
        return
    table = app.host_clip_table() if app.show_host else {}
    imgui.text_disabled("host plays")
    for a1 in al.pair.a1:
        imgui.same_line()
        if not app.show_host:
            imgui.text_disabled(str(a1))
            continue
        if a1 not in table:
            imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.60, 0.62, 0.68, 1.0))
            imgui.text("%d?" % a1)
            imgui.pop_style_color()
            if imgui.is_item_hovered():
                imgui.set_tooltip("the handler names a1 = %d and em%02d's pack has no "
                                  "slot %d. Either the action is unreachable, or its "
                                  "clip comes from somewhere this tool cannot see."
                                  % (a1, app.browsing_species, a1))
            continue
        on = app._host_clip == a1
        if imgui.small_button("%d%s##a%d" % (a1, " *" if on else "", a1)):
            app.play_host_clip(a1)
        if imgui.is_item_hovered():
            imgui.set_tooltip("%d frames%s — which of the %d runs depends on runtime "
                              "state" % (table[a1][0], ", loops" if table[a1][1] else "",
                                         len(al.pair.a1)))
    if app.show_host and table and not any(a in table for a in al.pair.a1):
        # 🔴 Say it. The reference is still on its default pose, and a viewer who
        # assumes that IS the action has been misled by the tool — the exact failure
        # the panel exists to prevent.
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.70, 0.20, 1.0))
        imgui.text_wrapped("! none of this action's clips is in em%02d's pack, so the "
                           "host beside you is showing its DEFAULT pose, not this "
                           "action. 112 of em75's 225 pairs are like this — see #34."
                           % app.browsing_species)
        imgui.pop_style_color()


def _finding_dots(imgui, al) -> None:
    """A one-line summary that stays visible when the findings are collapsed."""
    bits = [(f.code.replace("_", " ").lower(), f.level) for f in al.findings
            if f.level != "info"]
    if not bits:
        imgui.text_disabled("nothing to flag")
        return
    for i, (code, level) in enumerate(bits):
        if i:
            imgui.same_line()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(*_LEVEL_COLOR[level]))
        imgui.text(code)
        imgui.pop_style_color()


def _species_effects(imgui, app) -> None:
    """The species' framed spawn vocabulary — where the `@frame` numbers actually are.

    ⚠️ NOT attributed to this pair, and the file says so: em75's 30 framed sites all
    sit in routines hung off a species-byte switch no pair handler calls. Shown here
    because they are the only place the effect TIMING is legible, and hidden behind a
    header so they are never mistaken for this action's own.
    """
    fx = app.intel.framed_effects() if app.intel else []
    if not fx:
        return
    if not imgui.collapsing_header("%d framed effect site(s), species-wide" % len(fx)):
        return
    imgui.text_wrapped("! these are NOT attributed to any (main,sub) — they hang off a "
                       "species-byte switch no pair handler calls, so which action "
                       "fires them is not decidable offline. Bones are the host's.")
    rig = app.port_rig()
    for e in fx:
        imgui.bullet_text("effect %-3d  bone %-3s  frame %d" % (e.id, e.bone, e.frame))
        if imgui.is_item_hovered():
            imgui.set_tooltip("on YOUR rig: %s" % rig.describe(e.bone))



# --------------------------------------------------------------------------- #
# the part system (issue #10)
# --------------------------------------------------------------------------- #
def _part_swatch(imgui, part: int) -> None:
    """The gizmo's own colour, so the table and the viewport name the same thing."""
    from ..render.hitboxes import PART_COLORS

    r, g, b = PART_COLORS[part % len(PART_COLORS)]
    imgui.color_button("##sw%d" % part, imgui.ImVec4(r, g, b, 1.0),
                       imgui.ColorEditFlags_.no_tooltip.value,
                       imgui.ImVec2(12, 12))


def _volumes_now(app):
    """The volumes the panel is describing — whichever source is selected."""
    if app.parts_source == "port":
        sess = app.part_session
        return list(sess.volumes()) if sess is not None else []
    host = app.host_parts()
    return list(host.spheres()) if host is not None else []


def _part_table(imgui, app, host) -> None:
    """One row per accumulator slot: colour, name, what is attached, and isolate."""
    vols = _volumes_now(app)
    nb = app.scene.rig.n_bones
    real = [v for v in vols if not getattr(v, "is_marker", False)]
    imgui.text_disabled("%d volume(s) on %d bone(s) of %d%s"
                        % (len(real), len({v.bone for v in real}), nb,
                           "" if len(real) == len(vols)
                           else ", +%d walker marker(s)" % (len(vols) - len(real))))
    if app.parts_source == "host" and host is not None:
        if host.active is not None:
            imgui.text_disabled("the set species %s walks: 0x%08X (%d records). The "
                                "overlay holds %d; the rest are other species ids'."
                                % (",".join(str(x) for x in host.active.species)
                                   or "?", host.active.va, host.capacity,
                                   len(host.hurtboxes)))
        else:
            imgui.text_disabled("every hurtbox set in the overlay — rebuild "
                                "species/emNN.json (tools/em_intel.py --all) to "
                                "see only the one this species walks")
    if app._orphans:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.35, 0.35, 1.0))
        imgui.text_wrapped(
            plain("⚠ %d volume(s) name a bone this rig does not have (%s) and are "
                  "drawn NOWHERE. Bone indices belong to the rig that ships them."
                  % (len(app._orphans),
                     ", ".join(str(o.bone) for o in app._orphans[:6]))))
        imgui.pop_style_color()

    flags = (imgui.TableFlags_.borders_inner_h.value
             | imgui.TableFlags_.row_bg.value
             | imgui.TableFlags_.sizing_stretch_prop.value)
    if not imgui.begin_table("##parts", 6, flags, imgui.ImVec2(0.0, 170.0)):
        return
    for name, w in (("", 0.22), ("#", 0.22), ("name", 1.0), ("vols", 0.35),
                    ("bones", 0.9), ("row", 0.45)):
        imgui.table_setup_column(name, imgui.TableColumnFlags_.width_stretch.value, w)
    imgui.table_setup_scroll_freeze(0, 1)
    imgui.table_headers_row()

    by_part = {}
    for v in vols:
        by_part.setdefault(int(getattr(v, "part", 0) or 0) & 7, []).append(v)
    sess = app.part_session
    for i in range(8):
        mine = by_part.get(i, [])
        imgui.table_next_row()
        imgui.table_next_column()
        _part_swatch(imgui, i)
        imgui.table_next_column()
        sel = app.selected_part == i
        if imgui.selectable("%d##p%d" % (i, i), sel,
                            imgui.SelectableFlags_.span_all_columns.value)[0]:
            app.selected_part = None if sel else i
            app._part_name_buf = "" if sess is None else sess.name_of(i)
            if app.viewport is not None and app.viewport.hitboxes is not None:
                app.viewport.hitboxes.set_selected_part(app.selected_part)
                app.viewport.sync_hitbox_focus()
        imgui.table_next_column()
        nm = "" if sess is None else sess.name_of(i)
        if nm:
            imgui.text(nm)
        elif i == 0:
            imgui.text_disabled("unassigned")
        else:
            imgui.text_disabled("—")
        imgui.table_next_column()
        imgui.text(str(len(mine)) if mine else "·")
        imgui.table_next_column()
        bones = sorted({v.bone for v in mine})
        imgui.text(_bone_span(bones))
        if bones and imgui.is_item_hovered():
            imgui.set_tooltip(", ".join(str(b) for b in bones))
        imgui.table_next_column()
        rows = sorted({int(getattr(v, "hitzone_row", 0) or 0) for v in mine})
        imgui.text(",".join(str(r) for r in rows) if rows else "·")
        if len(rows) > 1 and imgui.is_item_hovered():
            imgui.set_tooltip(plain(
                "this part's volumes use DIFFERENT hitzone rows, so they take "
                "different percentages. Not an error — the Tigrex's wings do it."))
    imgui.end_table()

    if app.selected_part is not None and sess is not None:
        imgui.set_next_item_width(-90)
        _, app._part_name_buf = imgui.input_text(
            "##pname", app._part_name_buf, 32)
        imgui.same_line()
        if imgui.button("name %d" % app.selected_part):
            try:
                sess.name_part(app.selected_part, app._part_name_buf)
                app.status = "part %d = %s (unsaved)" % (app.selected_part,
                                                         app._part_name_buf)
            except Exception as e:                              # noqa: BLE001
                app.status = str(e)


def _bone_span(bones) -> str:
    """`10-14, 18` — a bone list at the width a table column actually has."""
    if not bones:
        return "·"
    out, start, prev = [], bones[0], bones[0]
    for b in bones[1:] + [None]:
        if b is not None and b == prev + 1:
            prev = b
            continue
        out.append(str(start) if start == prev else "%d-%d" % (start, prev))
        if b is not None:
            start = prev = b
    return ", ".join(out)


_SHAPES = ("sphere", "capsule")


def _volume_editor(imgui, app, host) -> None:
    """The port's volumes, one selectable row each, and the selected one's fields.

    This is the authoring half of issue #19: pick a sphere, make it obviously
    different (scale it, move it, keep only it), and ship it. Every change lands in
    `PartSession` and the viewport re-draws it on the live pose, so what the table
    says and what the gizmo shows are the same object.
    """
    sess = app.part_session
    if sess is None:
        imgui.text_disabled("no manifest, so no volumes to edit")
        return
    vols = sess.volumes()
    cap = sess.capacity
    if not vols:
        imgui.text_disabled("no [[hurtbox]] authored — adopt the host's below, or "
                            "add one")
        if imgui.button("add a sphere"):
            from ..manifest import Hurtbox
            i = sess.add_volume(Hurtbox(bone=1, radius=150.0, part=1, hitzone_row=0,
                                        offset=[0.0, 0.0, 0.0]))
            app.select_volume(i)
            app.sync_hitboxes()
        return

    over = sess.over_capacity
    head = "%d volume(s)" % len(vols)
    if cap is not None:
        head += ", %d fit in place" % cap
    imgui.text_disabled(head)
    if over:
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.35, 0.35, 1.0))
        imgui.text(plain("⚠ %d over — the runtime truncates" % over))
        imgui.pop_style_color()
    if app.selected_part is not None:
        imgui.same_line()
        _, app._only_selected_part = imgui.checkbox("only part %d" % app.selected_part,
                                                    app._only_selected_part)

    flags = (imgui.TableFlags_.borders_inner_h.value
             | imgui.TableFlags_.row_bg.value
             | imgui.TableFlags_.scroll_y.value
             | imgui.TableFlags_.sizing_stretch_prop.value)
    if imgui.begin_table("##vols", 7, flags, imgui.ImVec2(0.0, 150.0)):
        for name, w in (("#", 0.3), ("bone", 0.4), ("shape", 0.55), ("r", 0.5),
                        ("part", 0.4), ("row", 0.4), ("offset", 1.2)):
            imgui.table_setup_column(name, imgui.TableColumnFlags_.width_stretch.value, w)
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_headers_row()
        for i, v in enumerate(vols):
            part = int(v.part or 0) & 7
            if app._only_selected_part and app.selected_part is not None \
                    and part != app.selected_part:
                continue
            imgui.table_next_row()
            imgui.table_next_column()
            sel = app.selected_volume == i
            if imgui.selectable("%d##v%d" % (i, i), sel,
                                imgui.SelectableFlags_.span_all_columns.value)[0]:
                app.select_volume(None if sel else i)
            if sess.volume_changed(i) and imgui.is_item_hovered():
                imgui.set_tooltip("changed — not saved yet")
            imgui.table_next_column()
            if v.is_marker:
                imgui.text_disabled("0x%X" % v.bone)
                if imgui.is_item_hovered():
                    imgui.set_tooltip(plain("a walker MARKER, not a joint: 0x7D is "
                                            "the tail-sever skip. Drawn nowhere."))
            else:
                imgui.text(str(v.bone))
            imgui.table_next_column()
            imgui.text(v.shape[:4] if not v.is_marker else "mark")
            imgui.table_next_column()
            imgui.text("%g" % v.radius)
            imgui.table_next_column()
            _part_swatch(imgui, part)
            imgui.same_line()
            imgui.text(str(part))
            imgui.table_next_column()
            imgui.text(str(int(v.hitzone_row or 0)))
            imgui.table_next_column()
            o = v.offset or (0.0, 0.0, 0.0)
            imgui.text("%g %g %g" % tuple(o))
        imgui.end_table()

    i = app.selected_volume
    if i is None or not 0 <= i < len(vols):
        imgui.text_disabled("select a volume (here, or click its gizmo) to edit it")
        return
    v = vols[i]
    from ..manifest import ManifestError

    def stage(**fields):
        try:
            sess.edit_volume(i, **fields)
            app.sync_hitboxes()
            app.status = "volume %d: %s (unsaved)" % (
                i, ", ".join("%s=%s" % kv for kv in fields.items()))
        except ManifestError as e:
            app.status = str(e)

    imgui.text("volume %d%s" % (i, " — " + v.label if v.label else ""))
    imgui.set_next_item_width(70)
    ch, bone = imgui.input_int("bone##vb", int(v.bone), 1, 5)
    if ch:
        stage(bone=max(0, bone))
    if imgui.is_item_hovered():
        imgui.set_tooltip(plain("an index into THIS port's rig (%d joints). "
                                "Click a joint in the viewport to read its number."
                                % app.scene.rig.n_bones))
    imgui.same_line()
    imgui.set_next_item_width(60)
    ch, part = imgui.input_int("part##vp", int(v.part or 0), 1, 1)
    if ch:
        stage(part=max(0, min(7, part)))
    imgui.same_line()
    imgui.set_next_item_width(60)
    ch, row = imgui.input_int("row##vr", int(v.hitzone_row or 0), 1, 1)
    if ch:
        stage(hitzone_row=max(0, min(6, row)))
    if imgui.is_item_hovered():
        imgui.set_tooltip(plain("hitzone_row: which grid row's percentages this "
                                "volume takes. NOT the part."))
    imgui.same_line()
    imgui.set_next_item_width(90)
    ch, si = imgui.combo("##vshape", _SHAPES.index(v.shape), list(_SHAPES))
    if ch:
        stage(shape=_SHAPES[si], to=(list(v.to) if v.to else [0.0, 0.0, 200.0])
              if _SHAPES[si] == "capsule" else v.to)

    imgui.set_next_item_width(140)
    ch, r = imgui.drag_float("radius##vrad", float(v.radius), 1.0, 0.0, 5000.0, "%.1f")
    if ch:
        stage(radius=max(0.0, r))
    imgui.same_line()
    if imgui.button("x2"):
        stage(radius=v.radius * 2.0)
    imgui.same_line()
    if imgui.button("x3"):
        stage(radius=v.radius * 3.0)
    imgui.same_line()
    if imgui.button("x0.5"):
        stage(radius=v.radius * 0.5)

    imgui.set_next_item_width(230)
    ch, off = imgui.input_float3("offset##voff", list(v.offset or (0.0, 0.0, 0.0)),
                                 "%.1f")
    if ch:
        stage(offset=[float(x) for x in off])
    if v.is_capsule:
        imgui.set_next_item_width(230)
        ch, to = imgui.input_float3("to##vto", list(v.to or (0.0, 0.0, 0.0)), "%.1f")
        if ch:
            stage(to=[float(x) for x in to])
    imgui.text_disabled("flags 0x%X%s" % (v.flags, "" if not v.flags else
                                          "  (shipped as-is)"))

    if imgui.button("keep only this"):
        n = sess.keep_only(i)
        app.select_volume(0)
        app.sync_hitboxes()
        app.status = ("kept volume %d, dropped %d — one sphere: a hit lands there "
                      "or nowhere (unsaved)" % (i, n))
    if imgui.is_item_hovered():
        imgui.set_tooltip(plain("issue #19's experiment: with ONE volume left, where "
                                "a hit registers is the whole answer"))
    imgui.same_line()
    if imgui.button("delete"):
        sess.remove_volume(i)
        app.select_volume(None)
        app.sync_hitboxes()
        app.status = "deleted volume %d (unsaved)" % i
    imgui.same_line()
    if imgui.button("duplicate"):
        j = sess.add_volume(v)
        app.select_volume(j)
        app.sync_hitboxes()
        app.status = "volume %d duplicated as %d (unsaved)" % (i, j)


def _grid_view(imgui, app, host) -> None:
    """The damage grid: seven rows of ten, per state, editable when the port owns it."""
    sess = app.part_session
    own = [] if sess is None else sess.states()
    states = own if own else (host.states if host is not None else [])
    if not states:
        imgui.text_disabled("no damage grid — adopt the host's below")
        return
    editable = bool(own)

    imgui.text_disabled("damage grid — %d state(s)%s"
                        % (len(states), "" if editable else ", the HOST's (read only)"))
    if imgui.begin_tab_bar("##hzstates"):
        for i, st in enumerate(states):
            nm = getattr(st, "name", None) or "state %d" % i
            if imgui.begin_tab_item(" %s ##hz%d" % (nm, i))[0]:
                app._grid_state = i
                _grid_table(imgui, app, st, i, editable)
                imgui.end_tab_item()
        imgui.end_tab_bar()

    if host is not None and host.grid_note:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.70, 0.20, 1.0))
        imgui.text_wrapped(plain("⚠ " + host.grid_note))
        imgui.pop_style_color()
    imgui.text_wrapped(plain(
        "grid writes were proven live (2026-06-28: every byte 0xFF -> 411-damage "
        "hits); the VOLUMES are what #19 is still testing. A * marks a column whose "
        "NAME is inferred."))


def _grid_table(imgui, app, state, index: int, editable: bool) -> None:
    from ..manifest import HITZONE_COLUMNS

    host = app.host_parts()
    inferred = set() if host is None else set(host.inferred_columns())
    flags = (imgui.TableFlags_.borders.value
             | imgui.TableFlags_.row_bg.value
             | imgui.TableFlags_.sizing_fixed_fit.value
             | imgui.TableFlags_.scroll_x.value)
    if not imgui.begin_table("##grid%d" % index, 1 + len(HITZONE_COLUMNS), flags,
                             imgui.ImVec2(0.0, 190.0)):
        return
    imgui.table_setup_column("row")
    for c in HITZONE_COLUMNS:
        imgui.table_setup_column(c + ("*" if c in inferred else ""))
    imgui.table_setup_scroll_freeze(1, 1)
    imgui.table_headers_row()
    rows = state.rows
    sess = app.part_session
    for r, row in enumerate(rows):
        imgui.table_next_row()
        imgui.table_next_column()
        name = _row_owner(app, r)
        imgui.text("%d" % r)
        if name and imgui.is_item_hovered():
            imgui.set_tooltip("used by %s" % name)
        if editable:
            imgui.same_line()
            if imgui.small_button("255##max%d_%d" % (index, r)):
                try:
                    sess.fill_row(index, r, 255)
                    app.status = ("%s row %d: cut/impact/shot = 255 — the most a "
                                  "byte can say (unsaved)" % (state.name, r))
                except Exception as e:                          # noqa: BLE001
                    app.status = str(e)
            if imgui.is_item_hovered():
                imgui.set_tooltip(plain("cut, impact and shot to 255%% — every "
                                        "weapon class does the most the grid can "
                                        "express against this row"))
        for c, value in enumerate(row):
            imgui.table_next_column()
            if not editable:
                if value:
                    imgui.text("%d" % value)
                else:
                    imgui.text_disabled("0")
                continue
            imgui.set_next_item_width(38)
            ch, v = imgui.input_int("##g%d_%d_%d" % (index, r, c), int(value), 0, 0)
            if ch:
                try:
                    sess.set_hitzone(index, r, c, max(0, min(255, int(v))))
                    app.status = "%s row %d %s = %d (unsaved)" % (
                        state.name, r, HITZONE_COLUMNS[c], max(0, min(255, int(v))))
                except Exception as e:                          # noqa: BLE001
                    app.status = str(e)
    imgui.end_table()


def _row_owner(app, row: int) -> str:
    """Which named part reads this grid row — the join that makes the table legible."""
    sess = app.part_session
    if sess is None:
        return ""
    names = []
    for v in _volumes_now(app):
        if int(getattr(v, "hitzone_row", 0) or 0) != row:
            continue
        nm = sess.name_of(int(getattr(v, "part", 0) or 0) & 7)
        if nm and nm not in names:
            names.append(nm)
    return ", ".join(names)


def _part_actions(imgui, app, host, m) -> None:
    """Adopt, save, discard — and say what adopting actually costs."""
    sess = app.part_session
    if sess is None or m is None:
        imgui.text_disabled("this scene has no manifest, so there is nothing to "
                            "write parts into")
        return
    if host is not None and host.has_grid:
        if imgui.button("adopt the host's grid"):
            n = sess.adopt_grid(host.states)
            app.parts_source = "port"
            app.status = ("adopted %d grid state(s) — the port inherits these at "
                          "runtime either way (unsaved)" % n)
        imgui.same_line()
    if host is not None and host.spheres():
        if imgui.button("adopt the host's volumes"):
            got = sess.adopt_volumes(host.spheres(),
                                     source="em%02d" % (app.host_species or 0))
            app.parts_source = "port"
            app.sync_hitboxes()
            app.status = got.describe()
        if imgui.is_item_hovered():
            imgui.set_tooltip(plain(
                "⚠ the bone indices are the HOST's. This port ships its own rig, so "
                "a copied sphere lands on whatever joint happens to sit at that "
                "index — a starting point you can SEE, not a correct answer."))

    _export_buttons(imgui, app, m)

    pending = sess.pending
    if not pending:
        imgui.text_disabled("nothing staged")
        return
    if imgui.button("save %d to %s" % (pending, app.manifest_path.name
                                       if app.manifest_path else "the manifest")):
        try:
            app.status = sess.save() or "nothing to save"
            # re-read what actually landed: the session's staged copy and the file
            # can only be trusted to agree once the file has been parsed back.
            from ..manifest import load as _load
            app.scene.attach_manifest(_load(app.manifest_path))
            app._parts = None
            app.sync_hitboxes()
        except Exception as e:                                  # noqa: BLE001
            app.status = "%s: %s" % (type(e).__name__, e)
    imgui.same_line()
    if imgui.button("discard"):
        sess.discard()
        app.sync_hitboxes()
        app.status = "discarded the staged part edits"


def _export_buttons(imgui, app, m) -> None:
    """The manifest's tables -> the runtime's `<name>_hit.lua`, and onto the memstick.

    Exports what is SAVED: the file is the source of truth and the generated module
    says which file and which content it came from, so an unsaved edit would make
    the two disagree without a trace.
    """
    from .. import runtime as RT

    if app.manifest_path is None:
        return
    authored = bool(m.hurtboxes or m.hitzones or m.hitboxes or m.attacks)
    if not authored:
        imgui.text_disabled("nothing to export yet — save volumes, a grid, a hitbox "
                            "set or an attack record first")
        return

    def _export():
        return RT.export(m, capacity=RT.host_capacity(m),
                         attacks=RT.host_attack_tables(m))

    if imgui.button("export runtime table"):
        try:
            path = _export()
            app._hit_export = path
            app.status = "wrote %s (id %s)" % (path.name, RT.content_id(m))
        except Exception as e:                                  # noqa: BLE001
            app.status = "%s: %s" % (type(e).__name__, e)
    if imgui.is_item_hovered():
        imgui.set_tooltip(plain(
            "framework/prx/mods/lua_host/scripts/%s_hit.lua — the SAVED [[hurtbox]], "
            "[[hitzone]], [[hitbox]] and [[attack]] as the ONE P.hit() call "
            "mhfu_port.lua writes into the game: hurtboxes in place over the host set, "
            "the grid over the species blocks, each attack set in place through the "
            "overlay's pointer table, the levers by byte." % m.name))
    if RT.MEMSTICK_MODS.is_dir():
        imgui.same_line()
        if imgui.button("deploy to memstick"):
            try:
                path = app._hit_export or _export()
                app._hit_export = path
                dep = RT.deploy(path)
                app.status = ("deployed %s -> %s. A running game hot-reloads it; "
                              "a cold one loads it at boot"
                              % (dep.describe() if dep else path.name,
                                 RT.MEMSTICK_MODS) if dep else
                              "no memstick mods dir at %s" % RT.MEMSTICK_MODS)
            except Exception as e:                              # noqa: BLE001
                app.status = "%s: %s" % (type(e).__name__, e)
        if imgui.is_item_hovered():
            imgui.set_tooltip(plain("copy the module to %s — and mhfu_port.lua with it "
                                    "if the memstick's is behind: a stale library "
                                    "silently ignores fields it does not know"
                                    % RT.MEMSTICK_MODS))
    staged = (app.part_session is not None and app.part_session.pending) or (
        app._attacks is not None and app._attacks.pending)
    if staged:
        imgui.text_disabled("(exports the SAVED file — save first)")


# --------------------------------------------------------------------------- #
# the attack system (issue #33) — where he hits YOU
# --------------------------------------------------------------------------- #
def _set_swatch(imgui, set_index: int) -> None:
    """The gizmo's own colour for a set, so the table and the viewport agree."""
    from ..render.hitboxes import set_color

    r, g, b = set_color(set_index)
    imgui.color_button("##ss%d" % set_index, imgui.ImVec4(r, g, b, 1.0),
                       imgui.ColorEditFlags_.no_tooltip.value, imgui.ImVec2(12, 12))


def _attack_provenance(imgui, app, host) -> None:
    """One line saying where the join came from — em75's was walked live, the rest
    are an id range fitting a table, and a UI must not draw the two alike."""
    sp = "0x%08X" % host.spawner if host.spawner else "?"
    if host.join == "measured":
        imgui.text_disabled("spawner %s -> the %d-record table: MEASURED live (#33)"
                            % (sp, len(host.primary.attacks) if host.primary else 0))
    else:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.95, 0.75, 0.35, 1.0))
        imgui.text_wrapped(plain("spawner %s -> table: INFERRED (%s). Only em75's join "
                                 "was walked to the HP write; here the id range was "
                                 "matched to the biggest table." % (sp, host.join)))
        imgui.pop_style_color()
    if imgui.is_item_hovered() and host.join_provenance:
        imgui.set_tooltip(plain(host.join_provenance))
    off = host.id_offset(app.host_species) if app.host_species is not None else None
    if off is None:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.35, 0.35, 1.0))
        imgui.text_wrapped(plain("⚠ species %s shares this overlay but its id offset is "
                                 "unknown — a move's attack ids cannot be resolved to "
                                 "records" % app.host_species))
        imgui.pop_style_color()
    elif off:
        imgui.text_disabled("species %d uses record = handler id + %d" % (app.host_species, off))


def _moves_hitting(app, host, set_index: int):
    """`(declared move names, other pair count)` whose handler hits with this set."""
    si = app.host_intel()
    if si is None:
        return [], 0
    pairs = si.pairs_hitting_with(set_index, app.host_species)
    m = app.scene.manifest
    by_pair = {}
    if m is not None:
        for name, mv in m.moves.items():
            by_pair.setdefault((mv.main, mv.sub), []).append(name)
    names, others = [], 0
    for p in pairs:
        n = by_pair.get((p.main, p.sub))
        if n:
            names += n
        else:
            others += 1
    return names, others


def _sets_table(imgui, app, host) -> None:
    """One row per volume set: colour, index, the attacks that use it, the moves
    that spawn those, how many volumes (host / port), the bones, rigged or not."""
    from ..manifest import HITBOX_MARKER_BONES

    sess = app.attack_session if app.attacks_source == "port" else None
    pair_sets = app.pair_sets()
    if pair_sets:
        _, app._sets_of_move_only = imgui.checkbox(
            "only the sets (%d,%d) hits with: %s" % (app._pair[0], app._pair[1],
                                                     ", ".join(str(x) for x in pair_sets)),
            app._sets_of_move_only)
    elif app._pair is not None:
        imgui.text_disabled("(%d,%d) spawns no attack the static scan can see — a turn, "
                            "a roar, a walk; or a computed id" % app._pair)
    listed = sorted(pair_sets) if pair_sets and app._sets_of_move_only else [
        st.index for st in host.sets]
    if app.attacks_source == "port":
        authored = app.attack_session.sets() if app.attack_session is not None else []
        listed = sorted(set(listed) | set(authored)) if not (
            pair_sets and app._sets_of_move_only) else listed
    if app._attack_orphans:
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.35, 0.35, 1.0))
        imgui.text_wrapped(plain(
            "⚠ %d volume(s) name a bone this rig does not have (%s) and are drawn "
            "NOWHERE. Bone indices belong to the rig that ships them."
            % (len(app._attack_orphans),
               ", ".join(str(o.bone) for o in app._attack_orphans[:6]))))
        imgui.pop_style_color()
    imgui.text_disabled("%d set(s) listed of %d; %d attack record(s)"
                        % (len(listed), len(host.sets), len(host.attacks)))

    flags = (imgui.TableFlags_.borders_inner_h.value
             | imgui.TableFlags_.row_bg.value
             | imgui.TableFlags_.scroll_y.value
             | imgui.TableFlags_.sizing_stretch_prop.value)
    if not imgui.begin_table("##sets", 6, flags, imgui.ImVec2(0.0, 160.0)):
        return
    for name, w in (("", 0.2), ("set", 0.3), ("attacks", 0.9), ("moves", 0.9),
                    ("vols", 0.45), ("bones", 0.9)):
        imgui.table_setup_column(name, imgui.TableColumnFlags_.width_stretch.value, w)
    imgui.table_setup_scroll_freeze(0, 1)
    imgui.table_headers_row()
    for idx in listed:
        st = host.set(idx)
        imgui.table_next_row()
        imgui.table_next_column()
        _set_swatch(imgui, idx)
        imgui.table_next_column()
        sel = app.selected_set == idx
        if imgui.selectable("%d##s%d" % (idx, idx), sel,
                            imgui.SelectableFlags_.span_all_columns.value)[0]:
            app.select_set(None if sel else idx)
        if st is not None and not st.rigged and imgui.is_item_hovered():
            imgui.set_tooltip(plain("un-rigged: every record hangs on bone 126/127, the "
                                    "NODE's own position — projectile-shaped. Nothing "
                                    "here to re-align to a joint."))
        imgui.table_next_column()
        atks = host.attacks_using(idx)
        imgui.text(", ".join("%d(p%d)" % (a.id, a.power) for a in atks[:4])
                   + (" +%d" % (len(atks) - 4) if len(atks) > 4 else "") if atks else "·")
        if atks and imgui.is_item_hovered():
            imgui.set_tooltip(plain("\n".join("attack %d: %s" % (a.id, a.describe())
                                              for a in atks)))
        imgui.table_next_column()
        names, others = _moves_hitting(app, host, idx)
        txt = ", ".join(names[:3]) + (" +%d" % (len(names) - 3) if len(names) > 3 else "")
        if others:
            txt += (" " if txt else "") + "(%d pair%s)" % (others, "" if others == 1 else "s")
        imgui.text(txt or "·")
        imgui.table_next_column()
        n_host = 0 if st is None else st.capacity
        if sess is not None:
            n_port = len(sess.volumes_of(idx))
            over = sess.over_capacity(idx)
            if over:
                imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.35, 0.35, 1.0))
            imgui.text("%d/%d" % (n_port, n_host) if n_port else "·/%d" % n_host)
            if over:
                imgui.pop_style_color()
                if imgui.is_item_hovered():
                    imgui.set_tooltip(plain("%d more than fit in place — the runtime "
                                            "truncates" % over))
        else:
            imgui.text(str(n_host))
        imgui.table_next_column()
        if sess is not None and sess.volumes_of(idx):
            bones = sorted({h.bone for _, h in sess.volumes_of(idx)
                            if h.bone not in HITBOX_MARKER_BONES})
        else:
            bones = [] if st is None else st.bones
        imgui.text(_bone_span(bones) if bones else ("node" if st is not None
                                                     and not st.rigged else "·"))
        if bones and imgui.is_item_hovered():
            imgui.set_tooltip(", ".join(str(b) for b in bones))
    imgui.end_table()


def _attack_volume_editor(imgui, app, host) -> None:
    """The port's attack volumes — the selected set's, or all — one row each, and
    the selected one's fields. The hurtbox editor with `set` where `part` was."""
    sess = app.attack_session
    if sess is None:
        imgui.text_disabled("no manifest, so no hitboxes to edit")
        return
    from ..manifest import Hitbox, ManifestError

    vols = sess.volumes()
    st = app.selected_set
    if st is not None and not sess.volumes_of(st):
        hs = host.set(st)
        imgui.text_disabled("set %d: nothing authored — the host's %d record(s) stand"
                            % (st, 0 if hs is None else hs.capacity))
        if hs is not None and hs.spheres and imgui.button("adopt the host's set %d" % st):
            got = sess.adopt_set(st, hs.spheres, source="em%02d set %d"
                                 % (app.host_species or 0, st))
            app.sync_attacks()
            app.status = got.describe()
        if hs is not None and hs.spheres and imgui.is_item_hovered():
            imgui.set_tooltip(plain(
                "⚠ the bone indices are the HOST's. This port ships its own rig, so a "
                "copied sphere lands on whatever joint sits at that index — a starting "
                "point you can SEE, not a correct answer."))
        imgui.same_line()
        if imgui.button("add a sphere to set %d" % st):
            i = sess.add_volume(Hitbox(bone=1, radius=150.0, set=st,
                                       offset=[0.0, 0.0, 0.0]))
            app.select_attack_volume(i)
            app.sync_attacks()
        return
    if not vols:
        imgui.text_disabled("no [[hitbox]] authored — select a set above to adopt "
                            "the host's or add a sphere")
        return

    rows = sess.volumes_of(st) if st is not None else list(enumerate(vols))
    over = sess.over_capacity(st) if st is not None else sum(
        sess.over_capacity_all().values())
    head = ("set %d: %d volume(s)" % (st, len(rows)) if st is not None
            else "%d volume(s) over %d set(s)" % (len(vols), len(sess.sets())))
    cap = sess.capacities.get(st) if st is not None else None
    if cap is not None:
        head += ", %d fit in place" % cap
    imgui.text_disabled(head)
    if over:
        imgui.same_line()
        imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.98, 0.35, 0.35, 1.0))
        imgui.text(plain("⚠ %d over — the runtime truncates" % over))
        imgui.pop_style_color()

    flags = (imgui.TableFlags_.borders_inner_h.value
             | imgui.TableFlags_.row_bg.value
             | imgui.TableFlags_.scroll_y.value
             | imgui.TableFlags_.sizing_stretch_prop.value)
    if imgui.begin_table("##avols", 6, flags, imgui.ImVec2(0.0, 140.0)):
        for name, w in (("#", 0.3), ("set", 0.35), ("bone", 0.4), ("shape", 0.55),
                        ("r", 0.5), ("offset", 1.2)):
            imgui.table_setup_column(name, imgui.TableColumnFlags_.width_stretch.value, w)
        imgui.table_setup_scroll_freeze(0, 1)
        imgui.table_headers_row()
        for i, v in rows:
            imgui.table_next_row()
            imgui.table_next_column()
            sel = app.selected_attack_volume == i
            if imgui.selectable("%d##av%d" % (i, i), sel,
                                imgui.SelectableFlags_.span_all_columns.value)[0]:
                app.select_attack_volume(None if sel else i)
            if sess.volume_changed(i) and imgui.is_item_hovered():
                imgui.set_tooltip("changed — not saved yet")
            imgui.table_next_column()
            _set_swatch(imgui, v.set)
            imgui.same_line()
            imgui.text(str(v.set))
            imgui.table_next_column()
            if v.is_node_space:
                imgui.text_disabled("node")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(plain("bone %d: at the NODE's own position — the "
                                            "attacker's origin for a body attack, a "
                                            "projectile's for a thrown one. Drawn at "
                                            "the origin." % v.bone))
            elif v.is_marker:
                imgui.text_disabled("0x7D")
                if imgui.is_item_hovered():
                    imgui.set_tooltip(plain("a JOINER the walker hands to 0x09C386A0; "
                                            "no geometry of its own. Drawn nowhere."))
            else:
                imgui.text(str(v.bone))
            imgui.table_next_column()
            imgui.text(v.shape[:4])
            imgui.table_next_column()
            imgui.text("%g" % v.radius)
            imgui.table_next_column()
            o = v.offset or (0.0, 0.0, 0.0)
            imgui.text("%g %g %g" % tuple(o))
        imgui.end_table()

    i = app.selected_attack_volume
    if i is None or not 0 <= i < len(vols):
        imgui.text_disabled("select a volume (here, or click its gizmo) to edit it")
        if st is not None:
            if imgui.button("add a sphere to set %d" % st):
                j = sess.add_volume(Hitbox(bone=1, radius=150.0, set=st,
                                           offset=[0.0, 0.0, 0.0]))
                app.select_attack_volume(j)
                app.sync_attacks()
            imgui.same_line()
            hs = host.set(st)
            if hs is not None and hs.spheres and imgui.button("re-adopt the host's set %d" % st):
                got = sess.adopt_set(st, hs.spheres, source="em%02d set %d"
                                     % (app.host_species or 0, st))
                app.sync_attacks()
                app.status = got.describe()
        return
    v = vols[i]

    def stage(**fields):
        try:
            sess.edit_volume(i, **fields)
            app.sync_attacks()
            app.status = "hitbox %d: %s (unsaved)" % (
                i, ", ".join("%s=%s" % kv for kv in fields.items()))
        except ManifestError as e:
            app.status = str(e)

    imgui.text("hitbox %d%s" % (i, " — " + v.label if v.label else ""))
    imgui.set_next_item_width(70)
    ch, bone = imgui.input_int("bone##ab", int(v.bone), 1, 5)
    if ch:
        stage(bone=max(0, bone))
    if imgui.is_item_hovered():
        imgui.set_tooltip(plain("an index into THIS port's rig (%d joints); 126/127 = the "
                                "node's own position. Click a joint in the viewport to "
                                "read its number." % app.scene.rig.n_bones))
    imgui.same_line()
    imgui.set_next_item_width(60)
    ch, sv = imgui.input_int("set##as", int(v.set), 1, 1)
    if ch:
        stage(set=max(0, min(len(host.sets) - 1, sv)))
    if imgui.is_item_hovered():
        imgui.set_tooltip(plain("which of the host's %d volume sets this record ships "
                                "in — the attack record's +0x0A picks the set"
                                % len(host.sets)))
    imgui.same_line()
    imgui.set_next_item_width(90)
    ch, si_ = imgui.combo("##ashape", _SHAPES.index(v.shape), list(_SHAPES))
    if ch:
        stage(shape=_SHAPES[si_], to=(list(v.to) if v.to else [0.0, 0.0, 200.0])
              if _SHAPES[si_] == "capsule" else v.to)

    imgui.set_next_item_width(140)
    ch, r = imgui.drag_float("radius##arad", float(v.radius), 1.0, 0.0, 5000.0, "%.1f")
    if ch:
        stage(radius=max(0.0, r))
    imgui.same_line()
    if imgui.button("x2##a"):
        stage(radius=v.radius * 2.0)
    imgui.same_line()
    if imgui.button("x3##a"):
        stage(radius=v.radius * 3.0)
    imgui.same_line()
    if imgui.button("x0.5##a"):
        stage(radius=v.radius * 0.5)

    imgui.set_next_item_width(230)
    ch, off = imgui.input_float3("offset##aoff", list(v.offset or (0.0, 0.0, 0.0)), "%.1f")
    if ch:
        stage(offset=[float(x) for x in off])
    if v.is_capsule:
        imgui.set_next_item_width(230)
        ch, to = imgui.input_float3("to##ato", list(v.to or (0.0, 0.0, 0.0)), "%.1f")
        if ch:
            stage(to=[float(x) for x in to])
    imgui.set_next_item_width(-120)
    ch, app._attack_label_buf = imgui.input_text("##alabel", v.label or "", 48)
    if ch:
        stage(label=app._attack_label_buf)
    imgui.same_line()
    imgui.text_disabled("flags 0x%X" % v.flags)

    if imgui.button("keep only this in set %d" % v.set):
        n = sess.keep_only(i)
        app.select_attack_volume(sess.volumes_of(v.set)[0][0])
        app.sync_attacks()
        app.status = ("kept hitbox %d, dropped %d from set %d — one sphere: the "
                      "attack lands there or nowhere (unsaved)" % (i, n, v.set))
    if imgui.is_item_hovered():
        imgui.set_tooltip(plain("the #33 experiment, per attack: with ONE volume left in "
                                "the set, where the blow lands is the whole answer. The "
                                "other sets are other attacks and stay."))
    imgui.same_line()
    if imgui.button("delete##a"):
        sess.remove_volume(i)
        app.select_attack_volume(None)
        app.sync_attacks()
        app.status = "deleted hitbox %d (unsaved)" % i
    imgui.same_line()
    if imgui.button("duplicate##a"):
        j = sess.add_volume(v)
        app.select_attack_volume(j)
        app.sync_attacks()
        app.status = "hitbox %d duplicated as %d (unsaved)" % (i, j)


def _attack_levers(imgui, app, host) -> None:
    """The records that use the selected set (or the selected pair's): power,
    element, volume — the three MEASURED levers, editable on the port; the host's
    byte shown where the port says nothing."""
    from ..manifest import ManifestError

    hp = app.host_pair()
    if app.selected_set is not None:
        recs = host.attacks_using(app.selected_set)
        title = "attack records using set %d" % app.selected_set
    elif hp is not None:
        recs = host.records_for(hp.attack_ids, app.host_species)
        title = "attack records (%d,%d) spawns" % app._pair
    else:
        imgui.text_disabled("select a set or a pair to see its attack records")
        return
    if not recs:
        imgui.text_disabled(title + ": none")
        return
    imgui.text_disabled(title + " — power / element gate / set. Only these three are "
                        "decoded (#33).")
    sess = app.attack_session if app.attacks_source == "port" else None
    flags = (imgui.TableFlags_.borders_inner_h.value
             | imgui.TableFlags_.row_bg.value
             | imgui.TableFlags_.sizing_stretch_prop.value)
    if not imgui.begin_table("##levers", 5, flags):
        return
    for name, w in (("id", 0.3), ("power", 0.6), ("element", 0.6), ("set", 0.6),
                    ("", 0.5)):
        imgui.table_setup_column(name, imgui.TableColumnFlags_.width_stretch.value, w)
    imgui.table_headers_row()
    for a in recs:
        mine = None if sess is None else sess.attack(a.id)
        imgui.table_next_row()
        imgui.table_next_column()
        imgui.text(str(a.id))
        if imgui.is_item_hovered():
            imgui.set_tooltip(plain("record 0x%08X: kind %d, angle %d, tag 0x%02X — "
                                    "located, not decoded" % (a.va, a.kind, a.angle, a.tag)))
        if sess is None:
            imgui.table_next_column()
            imgui.text(str(a.power))
            imgui.table_next_column()
            imgui.text("0x%02X" % a.element)
            imgui.table_next_column()
            imgui.text(str(a.volume))
            imgui.table_next_column()
            continue

        def lever(key, cur_host, fmt_hex=False):
            val = getattr(mine, key) if mine is not None else None
            imgui.set_next_item_width(-1)
            shown = cur_host if val is None else val
            if fmt_hex:
                ch, txt = imgui.input_text("##%s%d" % (key, a.id), "0x%02X" % shown, 8)
                if ch:
                    try:
                        nv = int(txt, 0)
                    except ValueError:
                        return
                else:
                    return
            else:
                ch, nv = imgui.input_int("##%s%d" % (key, a.id), int(shown), 1, 10)
                if not ch:
                    return
            try:
                nv = max(0, min(255, int(nv)))
                sess.set_attack(a.id, **{key: None if nv == cur_host else nv})
                app.status = "attack %d %s = %d%s (unsaved)" % (
                    a.id, key, nv, " = the host's, so the block says nothing"
                    if nv == cur_host else "")
            except ManifestError as e:
                app.status = str(e)

        imgui.table_next_column()
        lever("power", a.power)
        imgui.table_next_column()
        lever("element", a.element, fmt_hex=True)
        imgui.table_next_column()
        lever("volume", a.volume)
        imgui.table_next_column()
        if mine is not None and not mine.is_empty:
            if imgui.small_button("host##r%d" % a.id):
                sess.clear_attack(a.id)
                app.status = "attack %d: back to the host's bytes (unsaved)" % a.id
            if imgui.is_item_hovered():
                imgui.set_tooltip("drop the port's block; the host's record stands")
        else:
            imgui.text_disabled("host")
    imgui.end_table()


def _attack_actions(imgui, app, host, m) -> None:
    """Adopt the move's sets, export, save, discard — and the shared-data caveat."""
    sess = app.attack_session
    if sess is None or m is None:
        imgui.text_disabled("this scene has no manifest, so there is nothing to write "
                            "hitboxes into")
        return
    ps = app.pair_sets()
    if ps and imgui.button("adopt the sets (%d,%d) hits with" % app._pair):
        n = 0
        for st in ps:
            hs = host.set(st)
            if hs is not None and hs.spheres:
                sess.adopt_set(st, hs.spheres, source="em%02d set %d"
                               % (app.host_species or 0, st))
                n += 1
        app.attacks_source = "port"
        app.sync_attacks()
        app.status = ("adopted %d set(s) from em%02d — the HOST's bone indices, on "
                      "this rig: a starting point you can see, not a correct answer "
                      "(unsaved)" % (n, app.host_species or 0))
    if ps and imgui.is_item_hovered():
        imgui.set_tooltip(plain("copy the host's records for every set this pair's "
                                "handler spawns, so you can move them onto the right "
                                "joints of THIS rig"))
    imgui.push_style_color(imgui.Col_.text, imgui.ImVec4(0.95, 0.75, 0.35, 1.0))
    imgui.text_wrapped(plain("🔴 the sets are SPECIES data in the overlay: with the port "
                             "REPLACING its host they are his alone; beside a native "
                             "em%02d they re-arm the native too. Proven live 2026-09-11: the "
                             "in-place write by RAM poke, the P.hit() path in a running "
                             "quest. ⚠ deploy syncs mhfu_port.lua too — a stale library "
                             "silently drops these tables." % (app.host_species or 0)))
    imgui.pop_style_color()

    _export_buttons(imgui, app, m)

    pending = sess.pending
    if not pending:
        imgui.text_disabled("nothing staged")
        return
    if imgui.button("save %d to %s##atk" % (pending, app.manifest_path.name
                                            if app.manifest_path else "the manifest")):
        try:
            app.status = sess.save() or "nothing to save"
            from ..manifest import load as _load
            app.scene.attach_manifest(_load(app.manifest_path))
            app._attacks = None
            app._parts = None
            app.sync_attacks()
        except Exception as e:                                  # noqa: BLE001
            app.status = "%s: %s" % (type(e).__name__, e)
    imgui.same_line()
    if imgui.button("discard##atk"):
        sess.discard()
        app.sync_attacks()
        app.status = "discarded the staged hitbox edits"


def _hits_with_row(imgui, al, app) -> None:
    """The action inspector's new row (#33): what this action actually hits with —
    the attack records its handler spawns and the volume sets they point at, with a
    button into the Hitboxes panel."""
    host = app.host_attacks()
    p = al.pair
    if host is None or p is None or not app.browsing_the_host:
        return
    if not p.attack_ids:
        imgui.text_disabled("hits with: nothing the static scan can see%s"
                            % (" (%d computed id%s)" % (p.attack_sites_computed,
                                                        "" if p.attack_sites_computed == 1
                                                        else "s")
                               if p.attack_sites_computed else ""))
        return
    recs = host.records_for(p.attack_ids, app.host_species)
    if not recs:
        imgui.text_disabled("hits with: attack id(s) %s — unresolvable for species %s "
                            "(no id offset known)"
                            % (",".join(str(x) for x in p.attack_ids), app.host_species))
        return
    sets = sorted({a.volume for a in recs})
    imgui.text("hits with:")
    imgui.same_line()
    imgui.text_wrapped("; ".join(
        "attack %d (power %d, elem 0x%02X) -> set %d%s"
        % (a.id, a.power, a.element, a.volume,
           "" if host.set(a.volume) is None else " [%s]"
           % (host.set(a.volume).describe()[:60]
              + ("…" if len(host.set(a.volume).describe()) > 60 else "")))
        for a in recs))
    for st in sets:
        if imgui.small_button("edit set %d in Hitboxes" % st):
            app.show_attacks = True
            app.attacks_source = "port" if (
                app.attack_session is not None and app.attack_session.volumes_of(st)
            ) else "host"
            app.select_set(st)
            app.sync_attacks()
        imgui.same_line()
    imgui.new_line()


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
