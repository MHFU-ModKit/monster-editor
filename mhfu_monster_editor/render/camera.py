"""The orbit camera and the 4x4s it produces — pure numpy, no GL, no window.

Kept free of GL on purpose: framing is the part that is easy to get wrong and cheap to
test, and a test that needs a graphics driver does not run in CI. Everything here is
ordinary arithmetic over ``(4,4)`` float64 arrays in the **row-vector-free** convention
(``M @ v``, translation in the last COLUMN) — the same convention
:mod:`mhfu_monster_editor.core.pose` already uses for the rig, so a joint matrix and a
view matrix compose without anybody transposing anything by feel.

🔴 **World space IS engine space.** MHFU's model space is X = flank, **Y = up**,
**+Z = the nose** — measured, not assumed: on the native Tigrex the chain off the body
fork runs to Z +244 (the head), the tail chain to Z −653, and both leg chains to
Y −298. That is right-handed with the animal facing +Z, which is exactly OpenGL's world
convention with the front view on the +Z axis. So there is **no conversion matrix
anywhere in this package** — a vertex out of `core` goes into a vertex buffer as it is.
(`blender_mhfu/importer.py` needs its ``(x, y, z) -> (x, -z, y)`` only because Blender
is Z-up.)

Consequently :attr:`OrbitCamera.yaw` ``0`` looks the animal in the face, and the named
views below match `blender_mhfu/render_port_views.py`'s ``DIRS`` — deliberately, so a
picture from this tool and a picture from the Blender oracle can be laid side by side.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np

#: world up. Y, because the engine's Y is up (see the module docstring).
UP = np.array([0.0, 1.0, 0.0])

#: ``name -> (yaw, pitch)`` in degrees, mirroring `render_port_views.DIRS` so the two
#: renderers can be compared frame for frame. Yaw is around Y from +Z; pitch lifts.
VIEWS: Dict[str, Tuple[float, float]] = {
    "front": (0.0, 8.0),        # on +Z, looking at the nose
    "back": (180.0, 8.0),
    "side": (90.0, 7.0),        # on +X, the profile
    "other_side": (-90.0, 7.0),
    "three": (40.0, 20.0),      # the three-quarter that reads best as a thumbnail
    "top": (0.0, 80.0),
}

#: how much wider than the subject the framing is. 1.0 touches the bounding sphere.
FRAME_MARGIN = 1.15

#: pitch never reaches the pole: at exactly ±90° the up vector and the view direction
#: are parallel and `look_at` divides by zero.
MAX_PITCH = 89.0


# --------------------------------------------------------------------------- #
# the 4x4s
# --------------------------------------------------------------------------- #
def look_at(eye: Sequence[float], target: Sequence[float],
            up: Sequence[float] = UP) -> np.ndarray:
    """Right-handed view matrix. ``M @ v`` takes a world point to eye space."""
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    fwd = target - eye
    n = np.linalg.norm(fwd)
    if n < 1e-12:
        raise ValueError("look_at: the eye is at the target")
    fwd /= n
    up = np.asarray(up, dtype=np.float64)
    right = np.cross(fwd, up)
    rn = np.linalg.norm(right)
    if rn < 1e-9:
        # looking straight along `up` — pick any perpendicular rather than blow up.
        right = np.cross(fwd, np.array([0.0, 0.0, 1.0]))
        rn = np.linalg.norm(right)
        if rn < 1e-9:
            right = np.array([1.0, 0.0, 0.0])
            rn = 1.0
    right /= rn
    true_up = np.cross(right, fwd)
    m = np.eye(4)
    m[0, :3], m[1, :3], m[2, :3] = right, true_up, -fwd
    m[:3, 3] = -(m[:3, :3] @ eye)
    return m


def perspective(fov_y_deg: float, aspect: float, near: float, far: float) -> np.ndarray:
    """Right-handed projection onto OpenGL's ``z in [-1, 1]`` clip volume."""
    if aspect <= 0:
        raise ValueError("perspective: aspect must be > 0, got %r" % aspect)
    if not 0 < near < far:
        raise ValueError("perspective: need 0 < near < far, got %r/%r" % (near, far))
    f = 1.0 / math.tan(math.radians(fov_y_deg) * 0.5)
    m = np.zeros((4, 4))
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = (2.0 * far * near) / (near - far)
    m[3, 2] = -1.0
    return m


def gl_bytes(m: np.ndarray) -> bytes:
    """A row-major ``(4,4)`` as the column-major float32 GL wants in a uniform.

    ``m.T`` is columns-first, and `astype` materialises that order — the one line
    where a silent transpose would put the model somewhere off screen.
    """
    return np.ascontiguousarray(m.T, dtype="f4").tobytes()


# --------------------------------------------------------------------------- #
# bounds
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Bounds:
    """An axis-aligned box, and the sphere a camera actually frames."""
    lo: np.ndarray
    hi: np.ndarray

    @classmethod
    def of(cls, points: np.ndarray) -> "Bounds":
        p = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        if not len(p):
            return cls(np.zeros(3), np.zeros(3))
        return cls(p.min(axis=0), p.max(axis=0))

    @classmethod
    def union(cls, *parts: "Bounds") -> "Bounds":
        real = [b for b in parts if b is not None]
        if not real:
            return cls(np.zeros(3), np.zeros(3))
        return cls(np.min([b.lo for b in real], axis=0),
                   np.max([b.hi for b in real], axis=0))

    @property
    def center(self) -> np.ndarray:
        return (self.lo + self.hi) * 0.5

    @property
    def size(self) -> np.ndarray:
        return self.hi - self.lo

    @property
    def radius(self) -> float:
        """Half the diagonal — the bounding SPHERE, which is what framing needs."""
        return float(np.linalg.norm(self.size) * 0.5)

    def __repr__(self) -> str:
        return "<Bounds %s..%s r=%.1f>" % (np.round(self.lo, 1), np.round(self.hi, 1),
                                           self.radius)


# --------------------------------------------------------------------------- #
# the camera
# --------------------------------------------------------------------------- #
class OrbitCamera:
    """Target, distance, yaw, pitch — the four numbers an orbit camera really is.

    Angles are DEGREES at the API and radians nowhere the caller can see. Yaw turns
    around world Y measured from +Z, so ``yaw=0`` is the front view; pitch lifts the
    eye and is clamped short of the pole.
    """

    def __init__(self, target: Sequence[float] = (0.0, 0.0, 0.0),
                 distance: float = 10.0, yaw: float = VIEWS["three"][0],
                 pitch: float = VIEWS["three"][1], fov: float = 40.0) -> None:
        self.target = np.asarray(target, dtype=np.float64).copy()
        self.distance = float(distance)
        self.yaw = float(yaw)
        self._pitch = 0.0
        self.pitch = float(pitch)
        self.fov = float(fov)
        #: set by :meth:`frame`; keeps near/far sane as the user dollies in and out.
        self._scale = float(distance)

    # ---- state -------------------------------------------------------- #
    @property
    def pitch(self) -> float:
        return self._pitch

    @pitch.setter
    def pitch(self, v: float) -> None:
        self._pitch = max(-MAX_PITCH, min(MAX_PITCH, float(v)))

    @property
    def eye(self) -> np.ndarray:
        """Where the camera is, in world space."""
        y, p = math.radians(self.yaw), math.radians(self._pitch)
        cp = math.cos(p)
        d = np.array([cp * math.sin(y), math.sin(p), cp * math.cos(y)])
        return self.target + d * self.distance

    @property
    def near(self) -> float:
        return max(self._scale * 1e-3, 1e-4)

    @property
    def far(self) -> float:
        return max(self._scale * 40.0, self.distance * 4.0)

    # ---- framing ------------------------------------------------------ #
    def frame(self, bounds: Bounds, margin: float = FRAME_MARGIN) -> "OrbitCamera":
        """Point at ``bounds`` and back off until the bounding sphere fits.

        Uses the vertical field of view, so a window wider than it is tall always
        fits too. A degenerate subject (one point) still gets a workable distance
        rather than 0, which would put the eye inside the near plane.
        """
        self.target = bounds.center.copy()
        r = bounds.radius or 1.0
        self.distance = r * margin / math.sin(math.radians(self.fov) * 0.5)
        self._scale = self.distance
        return self

    def look(self, name: str) -> "OrbitCamera":
        """Snap to one of :data:`VIEWS` without changing target or distance."""
        if name not in VIEWS:
            raise KeyError("no view %r (have: %s)" % (name, ", ".join(sorted(VIEWS))))
        self.yaw, self.pitch = VIEWS[name]
        return self

    # ---- interaction -------------------------------------------------- #
    def orbit(self, dx: float, dy: float, speed: float = 0.4) -> None:
        """Drag in pixels -> degrees. Pitch clamps; yaw wraps."""
        self.yaw = (self.yaw - dx * speed) % 360.0
        self.pitch = self._pitch + dy * speed

    def pan(self, dx: float, dy: float, viewport_h: int) -> None:
        """Slide the target in the camera's own plane, at the target's depth.

        Scaled so a pixel of drag moves the target by a pixel of subject: the visible
        height at the target is ``2 * distance * tan(fov/2)``.
        """
        if viewport_h <= 0:
            return
        world_per_px = (2.0 * self.distance
                        * math.tan(math.radians(self.fov) * 0.5) / viewport_h)
        v = self.view
        right, up = v[0, :3], v[1, :3]
        self.target += (-dx * right + dy * up) * world_per_px

    def dolly(self, ticks: float, rate: float = 1.1) -> None:
        """Scroll -> multiplicative zoom, clamped so it cannot reach the target."""
        self.distance = max(self._scale * 1e-3, self.distance * (rate ** -ticks))

    # ---- matrices ----------------------------------------------------- #
    @property
    def view(self) -> np.ndarray:
        return look_at(self.eye, self.target, UP)

    def projection(self, aspect: float) -> np.ndarray:
        return perspective(self.fov, aspect, self.near, self.far)

    def mvp(self, aspect: float, model: Optional[np.ndarray] = None) -> np.ndarray:
        m = self.projection(aspect) @ self.view
        return m if model is None else m @ model

    def __repr__(self) -> str:
        return ("<OrbitCamera target=%s d=%.1f yaw=%.1f pitch=%.1f fov=%.0f>"
                % (np.round(self.target, 1), self.distance, self.yaw, self._pitch,
                   self.fov))
