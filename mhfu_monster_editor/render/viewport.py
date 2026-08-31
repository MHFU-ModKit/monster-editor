"""One frame of the 3D view — the single function the window and ``--headless`` share.

    vp = Viewport(ctx, (1280, 800))
    vp.set_scene(scene)                 # frames the camera on the subject
    vp.draw()                           # -> the resolved texture
    vp.target.save("out.png")

There is no second renderer. The app calls :meth:`draw` every frame and hands
:attr:`Target.texture` to ``imgui.image``; ``--headless`` calls the same
:meth:`draw` once and saves. That is the whole of issue #5's layering rule, and it is
what lets a regression image (issue #13) be evidence about the actual viewport rather
than about a lookalike.

What it draws, in order: the **skinned mesh** with its own TMH textures
(:mod:`.mesh`), the **bone overlay** with the body fork and the leading-origin chain
called out (:mod:`.skeleton`), and then the reference layers — bind-pose points, the
bounds box, the world axes and the ground (:mod:`.overlay`). Every one of them is a
plain attribute the UI binds a checkbox to.

🔴 It opens **posed, never at bind**: at bind every joint is at bind, so a mismapped
rig and a correct one produce the same picture, and the one thing a viewport exists to
catch would be invisible. → :func:`.mesh.default_pose`
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .camera import Bounds, OrbitCamera
from .mesh import (ISOLATE_HIDE, ISOLATE_OFF, ISOLATE_ONLY, MODE_FLAT, MODE_TEXTURED,
                   MODE_VGROUP, MODES, SkinnedMesh, default_pose)
from .overlay import (Ground, Lines, axes_geometry, bounds_geometry,
                      point_cloud_geometry)
from .skeleton import SkeletonOverlay
from .target import SAMPLES, Target

#: the viewport background. Dark, low-chroma: a monster is grey-brown and a bone
#: overlay is saturated, so both have to stay legible against it.
BACKGROUND = (0.102, 0.110, 0.129, 1.0)


def _bound_framebuffer(ctx):
    """The framebuffer to restore after drawing, or ``None`` when there is none.

    Two cases that both look like "just read ``ctx.fbo``" and are not:

    * ⚠️ ``ctx.fbo`` keeps referring to a framebuffer after it has been RELEASED, with
      its handle swapped for an ``InvalidObject``; calling ``use()`` on that raises.
      It happens whenever the previous owner of the binding has been torn down, which
      in this package is any second :class:`Viewport` or :class:`Target` in one process.
    * ⚠️ a **standalone (headless) context has no default framebuffer at all** —
      ``ctx.screen`` is ``None`` there, so there is nothing to fall back to and nothing
      that needs restoring. Only a windowed context has a back buffer to protect.
    """
    import moderngl

    fbo = ctx.fbo
    if fbo is not None and not isinstance(getattr(fbo, "mglo", None),
                                          moderngl.InvalidObject):
        return fbo
    return ctx.screen           # None in a standalone context


def _restore(fbo) -> None:
    if fbo is not None:
        fbo.use()


def scene_bounds(scene) -> Bounds:
    """The subject's extent at BIND — geometry if there is any, else the joints.

    A stable scale: it does not move while a clip plays, so the ground cell and the
    joint-marker size can be derived from it once. It is NOT what the camera frames —
    a bind pose is a splayed T whose centre is nowhere near the standing animal, so
    `Viewport.bounds` reports the posed extent instead.
    """
    parts = [Bounds.of(g.positions) for g in scene.groups if g.n_vertices]
    if not parts:
        return Bounds.of(scene.rig.bind_joints)
    return Bounds.union(*parts)


class Viewport:
    """Camera + target + the shell's drawables, with a scene bound into them."""

    def __init__(self, ctx, size: Tuple[int, int] = (1280, 800),
                 samples: int = SAMPLES) -> None:
        self.ctx = ctx
        self.target = Target(ctx, size, samples)
        self.camera = OrbitCamera()
        self.scene = None
        #: the BIND extent — a stable scale for the ground and the fallback framing.
        self.bind_bounds = Bounds.of(np.zeros((1, 3)))
        self.background = BACKGROUND
        # toggles the UI binds checkboxes straight to.
        self.show_mesh = True
        self.show_skeleton = True
        #: draw the skeleton THROUGH the mesh. On by default: an overlay you cannot
        #: see is not an overlay, and "which joint owns this plate" is the question
        #: the bone view exists to answer.
        self.skeleton_xray = True
        self.show_ground = True
        self.show_axes = False
        self.show_bounds = False
        self.show_points = False
        self.wireframe = False
        self.point_size = 2.0

        self.mesh: Optional[SkinnedMesh] = None
        self.skeleton: Optional[SkeletonOverlay] = None
        #: the clip and frame the mesh is currently deformed to. ``clip is None`` = bind.
        self.clip = None
        self.frame = 0.0

        self._ground = Ground(ctx)
        self._axes = Lines(ctx)
        self._box = Lines(ctx)
        self._points = Lines(ctx, mode=ctx.POINTS)

    # ---- binding ------------------------------------------------------ #
    def set_scene(self, scene, *, frame_camera: bool = True, posed: bool = True) -> None:
        """Bind a :class:`~mhfu_monster_editor.core.scene.Scene` and size everything.

        🔴 ``posed`` defaults True and issue #6 says why: **a rest-pose render cannot
        validate skinning**. At bind every joint is at bind, so a mismapped rig and a
        correct one are the same picture. The viewport opens on a real frame of a real
        clip (:func:`~.mesh.default_pose`) unless a caller explicitly asks for bind.
        """
        self._release_scene()
        self.scene = scene
        self.bind_bounds = scene_bounds(scene)
        # the ground is fitted ONCE, off the bind extent: a cell size that changed as
        # a clip played would make the whole world appear to breathe.
        self._ground.fit(self.bind_bounds)
        self._axes.set(*axes_geometry(self.bind_bounds.radius * 0.6))
        pts = [g.positions for g in scene.groups if g.n_vertices]
        cloud = np.concatenate(pts) if pts else scene.rig.bind_joints
        self._points.set(*point_cloud_geometry(cloud))

        self.mesh = SkinnedMesh(self.ctx, scene)
        self.skeleton = SkeletonOverlay(self.ctx, scene)
        self.skeleton.joint_size = max(4.0, min(9.0, self.bind_bounds.radius / 160.0))
        if posed:
            clip, frame = default_pose(scene)
            self.set_pose(clip, frame)
        else:
            self.set_pose(None)
        if frame_camera:
            # 🔴 frame the POSED animal, not the bind extent. A bind pose is a splayed
            # T — its centre is nowhere near where the standing animal actually is, so
            # framing on it puts the subject in a corner. Framed once here and left
            # alone while a clip plays, or the camera would chase every frame.
            self.camera.frame(self.bounds).look("three")

    def set_pose(self, clip, frame: float = 0.0) -> None:
        """Deform the mesh and move the skeleton onto one frame. ``clip=None`` = bind."""
        if self.scene is None:
            return
        self.clip = clip
        self.frame = float(frame)
        if clip is None:
            pose = self.scene.bind_pose()
            driven = None
        else:
            pose = self.scene.pose(clip, frame)
            driven = self.scene.clip(clip).driven
        if self.mesh is not None:
            self.mesh.set_pose(None if clip is None else pose)
        if self.skeleton is not None:
            self.skeleton.set_positions(pose.joints)
            self.skeleton.set_driven(driven)

    # ---- highlighting ------------------------------------------------- #
    def tag_joints(self, joints) -> None:
        """Paint these joints' geometry red — `MHFU_VIEW_HILITE`."""
        if self.mesh is not None:
            self.mesh.tag_joints(joints)

    def select_joint(self, joint: Optional[int]) -> None:
        if self.skeleton is not None:
            self.skeleton.set_selected(joint)

    @property
    def selected_joint(self) -> Optional[int]:
        return self.skeleton.selected if self.skeleton is not None else None

    @property
    def bounds(self) -> Bounds:
        """The extent of what is on screen NOW — the posed mesh, else bind."""
        if self.mesh is not None and self.mesh.bounds.radius > 0:
            return self.mesh.bounds
        return self.bind_bounds

    def resize(self, size: Tuple[int, int]) -> bool:
        return self.target.resize(size)

    # ---- the frame ---------------------------------------------------- #
    def draw(self):
        """Render one frame into :attr:`target` and return its resolved texture.

        🔴 **Restores whatever framebuffer was bound on entry.** Binding an FBO is
        global GL state, and in the app this method runs *inside* the host's frame:
        leaving our offscreen target bound means imgui then renders the ENTIRE UI into
        it and the window's back buffer is never drawn to at all — a completely black
        window, while every offscreen assertion still passes because the FBO does have
        the right pixels in it. That is precisely what happened, and why
        `tests/test_render_headless.py` now pins the binding and `tests/test_ui_smoke.py`
        reads the real back buffer instead of the target.
        """
        ctx, t = self.ctx, self.target
        prev = _bound_framebuffer(ctx)
        try:
            self._draw(ctx, t)
        finally:
            # `use()` restores the viewport with the binding, so the host's next draw
            # is not silently clipped to our panel's rectangle either.
            _restore(prev)
        return t.texture

    def _draw(self, ctx, t) -> None:
        t.use()
        t.clear(self.background)
        ctx.enable(ctx.DEPTH_TEST | ctx.BLEND | ctx.PROGRAM_POINT_SIZE)
        ctx.blend_func = ctx.SRC_ALPHA, ctx.ONE_MINUS_SRC_ALPHA

        mvp = self.camera.mvp(t.aspect)
        # face culling stays OFF: a monster PAC's triangle winding is not consistent,
        # so culling drops whole plates. The shader's lambert is two-sided for the
        # same reason.
        if self.show_mesh and self.mesh is not None:
            self.mesh.render(mvp, wireframe=self.wireframe)
        if self.show_skeleton and self.skeleton is not None:
            if self.skeleton_xray:
                ctx.disable(ctx.DEPTH_TEST)
                self.skeleton.render(mvp)
                ctx.enable(ctx.DEPTH_TEST)
            else:
                self.skeleton.render(mvp)
        if self.show_points:
            self._points.point_size = self.point_size
            self._points.render(mvp)
        if self.show_bounds:
            self._box.set(*bounds_geometry(self.bounds))
            self._box.render(mvp)
        if self.show_axes:
            self._axes.render(mvp)
        if self.show_ground:
            # last, and depth-write off: the ground is transparent, so writing depth
            # would let it erase whatever the blend behind it still needs.
            ctx.depth_mask = False
            self._ground.render(mvp)
            ctx.depth_mask = True

        t.resolve()

    # ---- teardown ----------------------------------------------------- #
    def _release_scene(self) -> None:
        """Free the per-scene GL objects. Opening a second PAC must not leak them."""
        for name in ("mesh", "skeleton"):
            obj = getattr(self, name, None)
            if obj is not None:
                obj.release()
            setattr(self, name, None)

    def release(self) -> None:
        self._release_scene()
        for o in (self._ground, self._axes, self._box, self._points, self.target):
            o.release()

    def __enter__(self) -> "Viewport":
        return self

    def __exit__(self, *exc) -> None:
        self.release()

    def __repr__(self) -> str:
        name = self.scene.name if self.scene is not None else "no scene"
        return "<Viewport %s %dx%d %r>" % (name, *self.target.size, self.camera)


# --------------------------------------------------------------------------- #
# the headless one-shot
# --------------------------------------------------------------------------- #
def render_to_file(scene, path, *, size: Tuple[int, int] = (1280, 800),
                   view: str = "three", samples: int = SAMPLES,
                   configure=None):
    """Open a standalone context, draw ``scene`` once, save a PNG, tear it all down.

    ``configure(viewport)`` runs after the scene is bound and the camera framed — the
    hook a caller uses to pick a different angle or turn a layer off without this
    function growing a keyword for every toggle.
    """
    from .context import headless
    ctx = headless()
    try:
        with Viewport(ctx, size, samples) as vp:
            vp.set_scene(scene)
            if view:
                vp.camera.look(view)
            if configure is not None:
                configure(vp)
            vp.draw()
            return vp.target.save(path)
    finally:
        ctx.release()
