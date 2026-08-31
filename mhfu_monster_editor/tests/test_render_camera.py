"""The camera and its 4x4s. **No GL** — this must run on a machine with no driver.

Which is the point: framing is where a viewer goes wrong (subject off screen, model
upside down, the front view showing the tail) and none of those need a graphics card to
catch. The GL-dependent tests live in `test_render_headless.py` and skip themselves.

    venv/bin/python mhfu_monster_editor/tests/test_render_camera.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from mhfu_monster_editor.render.camera import (  # noqa: E402
    MAX_PITCH, UP, VIEWS, Bounds, OrbitCamera, gl_bytes, look_at, perspective)


def _project(m, p):
    """A world point through a 4x4 into normalised device coordinates."""
    v = m @ np.array([p[0], p[1], p[2], 1.0])
    return v[:3] / v[3]


def test_bounds():
    b = Bounds.of(np.array([[-1.0, -2, -3], [1.0, 2, 3]]))
    assert np.allclose(b.center, 0), b.center
    assert np.allclose(b.size, [2, 4, 6]), b.size
    assert abs(b.radius - math.sqrt(4 + 16 + 36) / 2) < 1e-12
    assert np.allclose(Bounds.of(np.zeros((0, 3))).size, 0), "empty must not explode"

    u = Bounds.union(Bounds.of(np.array([[0.0, 0, 0]])), b)
    assert np.allclose(u.lo, [-1, -2, -3]) and np.allclose(u.hi, [1, 2, 3])
    print("bounds            centre/size/radius/union ok")


def test_look_at_is_a_rigid_frame():
    m = look_at([3.0, 4.0, 5.0], [0.0, 0.0, 0.0])
    r = m[:3, :3]
    assert np.allclose(r @ r.T, np.eye(3), atol=1e-12), "rotation is not orthonormal"
    assert abs(np.linalg.det(r) - 1.0) < 1e-12, "view matrix mirrors the world"
    # the eye lands at the origin of eye space, and the target on -Z in front of it.
    assert np.allclose(_project(m, [3.0, 4.0, 5.0])[:3], 0, atol=1e-12)
    t = m @ np.array([0.0, 0, 0, 1])
    assert t[2] < 0, "the target must be down -Z in a right-handed view"
    print("look_at           orthonormal, right-handed, eye at origin")


def test_look_at_degenerate_up():
    """Straight down the up axis must pick a frame, not divide by zero."""
    m = look_at([0.0, 10.0, 0.0], [0.0, 0.0, 0.0], UP)
    assert np.all(np.isfinite(m)), m
    try:
        look_at([1.0, 1, 1], [1.0, 1, 1])
    except ValueError:
        pass
    else:
        raise AssertionError("eye == target should raise")
    print("look_at           pole and eye==target handled")


def test_perspective_clip_volume():
    near, far = 1.0, 100.0
    p = perspective(45.0, 16 / 9, near, far)
    assert abs(_project(p, [0.0, 0, -near])[2] + 1.0) < 1e-12, "near must map to -1"
    assert abs(_project(p, [0.0, 0, -far])[2] - 1.0) < 1e-12, "far must map to +1"
    # a point on the top edge of the frustum lands on y = +1 exactly.
    y = near * math.tan(math.radians(45.0) / 2)
    assert abs(_project(p, [0.0, y, -near])[1] - 1.0) < 1e-12
    for bad in ((45.0, 0.0, 1.0, 10.0), (45.0, 1.0, 0.0, 10.0), (45.0, 1.0, 5.0, 1.0)):
        try:
            perspective(*bad)
        except ValueError:
            continue
        raise AssertionError("perspective%r should have been refused" % (bad,))
    print("perspective       near/far/edges map exactly, bad input refused")


def test_gl_bytes_is_column_major():
    m = np.arange(16, dtype=np.float64).reshape(4, 4)
    got = np.frombuffer(gl_bytes(m), dtype="f4")
    assert np.allclose(got, m.T.reshape(-1)), "GL uniforms are column-major"
    print("gl_bytes          column-major, f32")


def test_yaw_zero_is_the_front():
    """🔴 The convention the whole package rests on: +Z is the NOSE.

    Measured on the native Tigrex — the chain off the body fork runs to Z +244 (head),
    the tail chain to Z −653. So the camera that sees the face sits on +Z, and
    `VIEWS["front"]` must put it there. Flip this and every render is mislabelled.
    """
    cam = OrbitCamera(target=(0, 0, 0), distance=10.0).look("front")
    assert cam.eye[2] > 9.0, cam.eye
    assert abs(cam.eye[0]) < 1e-9, cam.eye

    cam.look("side")
    assert cam.eye[0] > 9.0 and abs(cam.eye[2]) < 1e-9, cam.eye

    cam.look("back")
    assert cam.eye[2] < -9.0, cam.eye

    cam.look("top")
    assert cam.eye[1] > 9.0, cam.eye
    print("views             front=+Z(nose) side=+X back=-Z top=+Y")


def test_pitch_clamps_short_of_the_pole():
    cam = OrbitCamera()
    cam.pitch = 400.0
    assert cam.pitch == MAX_PITCH
    cam.pitch = -400.0
    assert cam.pitch == -MAX_PITCH
    assert np.all(np.isfinite(cam.view)), "a clamped pitch must still give a view"
    print("pitch             clamped to +/-%.0f, view stays finite" % MAX_PITCH)


def test_frame_puts_the_whole_subject_on_screen():
    """The assertion that matters: after `frame`, every corner is inside the frustum."""
    rng = np.random.default_rng(7)
    for trial in range(24):
        lo = rng.uniform(-2000, 0, 3)
        hi = lo + rng.uniform(1, 3000, 3)
        b = Bounds(lo, hi)
        cam = OrbitCamera(fov=float(rng.uniform(20, 70))).frame(b)
        for name in VIEWS:
            cam.look(name)
            for aspect in (0.6, 1.0, 2.4):
                mvp = cam.mvp(aspect)
                corners = np.array([[x, y, z] for x in (lo[0], hi[0])
                                    for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
                ndc = np.array([_project(mvp, c) for c in corners])
                assert np.all(np.abs(ndc[:, 1]) <= 1.0 + 1e-9), \
                    "trial %d %s: subject leaves the frame vertically\n%s" \
                    % (trial, name, ndc)
                assert np.all(ndc[:, 2] > -1.0) and np.all(ndc[:, 2] < 1.0), \
                    "trial %d %s: subject clipped by near/far\n%s" % (trial, name, ndc)
    print("frame             24 boxes x %d views x 3 aspects all inside the frustum"
          % len(VIEWS))


def test_frame_survives_a_degenerate_subject():
    cam = OrbitCamera().frame(Bounds(np.zeros(3), np.zeros(3)))
    assert cam.distance > 0 and np.isfinite(cam.near) and cam.near < cam.far
    print("frame             a single point still gives a usable distance")


def test_orbit_pan_dolly():
    # pitch 0, so the eye stays in the XZ plane and the yaw is readable off it.
    cam = OrbitCamera(target=(0, 0, 0), distance=100.0, yaw=0.0, pitch=0.0)
    cam.orbit(90.0 / 0.4, 0.0)                 # 90 degrees of yaw at the default speed
    assert abs(cam.eye[0] + 100.0) < 1e-6, cam.eye     # yaw decreases with +dx
    assert cam.yaw == cam.yaw % 360.0, "yaw must stay wrapped"

    # again pitch 0, so the camera's own up IS world +Y and the drag is readable.
    cam = OrbitCamera(target=(0, 0, 0), distance=100.0, yaw=0.0, pitch=0.0, fov=40.0)
    h = 800
    before = cam.target.copy()
    cam.pan(0.0, h / 2.0, h)      # half a screen up
    moved = cam.target - before
    expect = 100.0 * math.tan(math.radians(20.0))       # half the visible height
    assert abs(moved[1] - expect) < 1e-9, (moved, expect)
    assert abs(moved[0]) < 1e-9 and abs(moved[2]) < 1e-9, "pan must stay in-plane"

    # and tilted: a pan is always in the camera's plane, so it stays perpendicular to
    # the view direction and keeps its length whatever the pitch.
    tilted = OrbitCamera(target=(0, 0, 0), distance=100.0, fov=40.0).look("three")
    fwd = tilted.target - tilted.eye
    start = tilted.target.copy()
    tilted.pan(0.0, h / 2.0, h)
    step = tilted.target - start
    assert abs(np.linalg.norm(step) - expect) < 1e-9, (step, expect)
    assert abs(float(step @ fwd)) < 1e-6, "pan must not change the distance"

    d0 = cam.distance
    cam.dolly(1.0)
    assert cam.distance < d0
    cam.dolly(-1.0)
    assert abs(cam.distance - d0) < 1e-9, "dolly must be reversible"
    for _ in range(400):
        cam.dolly(1.0)
    assert cam.distance > 0, "dolly must never reach the target"
    print("interaction       orbit/pan/dolly exact, dolly clamped")


def main() -> int:
    for fn in (test_bounds, test_look_at_is_a_rigid_frame, test_look_at_degenerate_up,
               test_perspective_clip_volume, test_gl_bytes_is_column_major,
               test_yaw_zero_is_the_front, test_pitch_clamps_short_of_the_pole,
               test_frame_puts_the_whole_subject_on_screen,
               test_frame_survives_a_degenerate_subject, test_orbit_pan_dolly):
        fn()
    print("\ntest_render_camera: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
