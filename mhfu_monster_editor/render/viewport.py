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

What is drawn here is the SHELL's content — ground, axes, bounds, and the bind-pose
vertices as points. The skinned mesh, the textures and the bone overlay are issue #6
and slot in beside these.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .camera import Bounds, OrbitCamera
from .overlay import (Ground, Lines, axes_geometry, bounds_geometry,
                      point_cloud_geometry)
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

    Bind rather than posed on purpose: the framing must not jump about while a clip
    plays, and a bind box contains every reasonable pose of the same rig.
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
        self.bounds = Bounds.of(np.zeros((1, 3)))
        self.background = BACKGROUND
        # toggles the shell owns; the UI binds checkboxes straight to these.
        self.show_ground = True
        self.show_axes = True
        self.show_bounds = False
        self.show_points = True
        self.point_size = 2.0

        self._ground = Ground(ctx)
        self._axes = Lines(ctx)
        self._box = Lines(ctx)
        self._points = Lines(ctx, mode=ctx.POINTS)

    # ---- binding ------------------------------------------------------ #
    def set_scene(self, scene, *, frame_camera: bool = True) -> None:
        """Bind a :class:`~mhfu_monster_editor.core.scene.Scene` and size everything."""
        self.scene = scene
        self.bounds = scene_bounds(scene)
        self._ground.fit(self.bounds)
        self._axes.set(*axes_geometry(self.bounds.radius * 0.6))
        self._box.set(*bounds_geometry(self.bounds))
        pts = [g.positions for g in scene.groups if g.n_vertices]
        cloud = np.concatenate(pts) if pts else scene.rig.bind_joints
        self._points.set(*point_cloud_geometry(cloud))
        if frame_camera:
            self.camera.frame(self.bounds).look("three")

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
        if self.show_points:
            self._points.point_size = self.point_size
            self._points.render(mvp)
        if self.show_bounds:
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
    def release(self) -> None:
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
