"""The things in the viewport that are not the monster: the ground, the axes, points.

All of it is `GL_LINES` / `GL_POINTS` through one flat-colour program, plus a single
procedurally-shaded quad for the ground. Small, and deliberately separate from the
subject: issue #6 adds the skinned mesh beside these, it does not replace them.

The point cloud is here because the app shell has to prove something. A window that
opens on an empty grid cannot tell you whether the PAC parsed, whether the camera
framed it or whether world space came out the right way up — and a headless PNG of a
grid asserts nothing at all. Drawing the bind-pose vertices costs one buffer and
answers all three at a glance. It is **not** the viewport of issue #6: no skinning, no
textures, no bone overlay.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

from .camera import Bounds, gl_bytes
from .shaders import ground_program, line_program

#: engine axis -> colour. X flank, Y up, Z nose (see :mod:`.camera`).
AXIS_COLORS = ((0.85, 0.28, 0.32), (0.42, 0.78, 0.36), (0.32, 0.52, 0.90))


def _nice_cell(radius: float) -> float:
    """A 1/2/5 x 10^n cell that puts roughly ten minor lines across the subject."""
    if radius <= 0:
        return 1.0
    raw = radius / 5.0
    mag = 10.0 ** np.floor(np.log10(raw))
    return float(mag * min((1, 2, 5, 10), key=lambda m: abs(raw / mag - m)))


class Ground:
    """An infinite-looking ground plane at y=0, shaded in the fragment shader."""

    def __init__(self, ctx) -> None:
        self.ctx = ctx
        self.prog = ground_program(ctx)
        self.cell = 100.0
        self.extent = 10_000.0
        self._vbo = ctx.buffer(reserve=6 * 3 * 4, dynamic=True)
        self._vao = ctx.vertex_array(self.prog, [(self._vbo, "3f", "in_pos")])
        self._built = None

    def fit(self, bounds: Bounds) -> None:
        """Size the cell and the plane to the subject, once per scene."""
        self.cell = _nice_cell(bounds.radius)
        self.extent = max(bounds.radius * 12.0, self.cell * 40.0)
        self._build()

    def _build(self) -> None:
        if self._built == self.extent:
            return
        e = self.extent
        quad = np.array([(-e, 0, -e), (e, 0, -e), (e, 0, e),
                         (-e, 0, -e), (e, 0, e), (-e, 0, e)], dtype="f4")
        self._vbo.write(quad.tobytes())
        self._built = self.extent

    def render(self, mvp: np.ndarray) -> None:
        self._build()
        p = self.prog
        p["u_mvp"].write(gl_bytes(mvp))
        p["u_cell"].value = self.cell
        p["u_major"].value = 5.0
        p["u_extent"].value = self.extent
        p["u_minor_color"].value = (0.32, 0.34, 0.38)
        p["u_major_color"].value = (0.46, 0.48, 0.53)
        p["u_axis_x"].value = AXIS_COLORS[0]
        p["u_axis_z"].value = AXIS_COLORS[2]
        self._vao.render(mode=self.ctx.TRIANGLES)

    def release(self) -> None:
        for o in (self._vao, self._vbo):
            o.release()


class Lines:
    """A flat-coloured `GL_LINES` / `GL_POINTS` batch: ``(N,3)`` positions + RGBA."""

    def __init__(self, ctx, mode: Optional[int] = None) -> None:
        self.ctx = ctx
        self.prog = line_program(ctx)
        self.mode = ctx.LINES if mode is None else mode
        self.count = 0
        self.alpha = 1.0
        self.point_size = 1.0
        self._vbo = None
        self._vao = None

    def set(self, positions: np.ndarray, colors: np.ndarray) -> None:
        """Upload. ``colors`` is ``(N,4)`` float, or one RGBA broadcast to all."""
        pos = np.ascontiguousarray(positions, dtype="f4").reshape(-1, 3)
        col = np.asarray(colors, dtype="f4")
        if col.ndim == 1:
            col = np.tile(col, (len(pos), 1))
        col = np.ascontiguousarray(col, dtype="f4").reshape(-1, 4)
        if len(col) != len(pos):
            raise ValueError("%d positions but %d colours" % (len(pos), len(col)))
        data = np.concatenate([pos, col], axis=1).astype("f4")
        need = data.nbytes
        if self._vbo is None or self._vbo.size < need:
            self.release()
            self._vbo = self.ctx.buffer(reserve=max(need, 1), dynamic=True)
            self._vao = self.ctx.vertex_array(
                self.prog, [(self._vbo, "3f 4f", "in_pos", "in_color")])
        self._vbo.write(data.tobytes())
        self.count = len(pos)

    def render(self, mvp: np.ndarray) -> None:
        if not self.count or self._vao is None:
            return
        self.prog["u_mvp"].write(gl_bytes(mvp))
        self.prog["u_alpha"].value = float(self.alpha)
        self.prog["u_point_size"].value = float(self.point_size)
        self._vao.render(mode=self.mode, vertices=self.count)

    def release(self) -> None:
        for o in (self._vao, self._vbo):
            if o is not None:
                o.release()
        self._vao = self._vbo = None


def axes_geometry(length: float) -> Tuple[np.ndarray, np.ndarray]:
    """The three world axes from the origin: ``(6,3)`` positions, ``(6,4)`` colours."""
    pos, col = [], []
    for a in range(3):
        d = np.zeros(3)
        d[a] = length
        pos += [np.zeros(3), d]
        col += [(*AXIS_COLORS[a], 1.0)] * 2
    return np.array(pos, dtype="f4"), np.array(col, dtype="f4")


def bounds_geometry(b: Bounds, rgba=(0.55, 0.57, 0.62, 0.55)
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """The twelve edges of a box — the cheapest 'is it framed?' answer there is."""
    lo, hi = b.lo, b.hi
    c = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1])
                  for z in (lo[2], hi[2])])          # index = 4*ix + 2*iy + iz
    edges = [(0, 1), (2, 3), (4, 5), (6, 7),
             (0, 2), (1, 3), (4, 6), (5, 7),
             (0, 4), (1, 5), (2, 6), (3, 7)]
    pos = np.array([c[i] for e in edges for i in e], dtype="f4")
    return pos, np.tile(np.array(rgba, dtype="f4"), (len(pos), 1))


def point_cloud_geometry(positions: np.ndarray, *, up_axis: int = 1,
                         low=(0.30, 0.55, 0.85), high=(0.95, 0.80, 0.45)
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Vertices tinted by height, so the silhouette reads as a shape and not a blob."""
    p = np.ascontiguousarray(positions, dtype="f4").reshape(-1, 3)
    if not len(p):
        return p, np.zeros((0, 4), dtype="f4")
    h = p[:, up_axis]
    span = float(h.max() - h.min())
    t = ((h - h.min()) / span if span > 1e-6 else np.zeros(len(p)))[:, None]
    col = np.array(low, dtype="f4") * (1 - t) + np.array(high, dtype="f4") * t
    return p, np.concatenate([col, np.ones((len(p), 1), dtype="f4")],
                             axis=1).astype("f4")
