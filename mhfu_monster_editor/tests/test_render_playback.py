"""Engine-accurate playback, root motion, and #7's "every clip plays cleanly".

The transport is pure arithmetic, so most of this runs with no driver: the gate, the
loop wrap, the wall-clock identity, the root strip. The GL half then does what issue #7
actually asks — plays **every** clip of both the native Tigrex and the built Zinogre
port, and times it.

    venv/bin/python mhfu_monster_editor/tests/test_render_playback.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(_ROOT))

from mhfu_monster_editor.render.playback import (  # noqa: E402
    DEFAULT_SPEED, GAME_HZ, Playback, pose_at, root_joints, root_travel,
    travel_joints, wall_clock)

TIGREX = _ROOT / "workspace" / "extracted" / "data_files" / "file_06185.bin"
ZINOGRE = _ROOT / "tmp" / "zinogre_v10.bin"


class _Clip:
    def __init__(self, frames, loop, slot=1):
        self.frames, self.loop, self.slot = frames, loop, slot


# --------------------------------------------------------------------------- #
# the transport — no GL, no files
# --------------------------------------------------------------------------- #
def test_the_gate_is_tested_before_the_step():
    """🔴 ``phase + speed <= end``, checked BEFORE advancing (compare at 0x088638E0).

    So a clip never shows a frame past ``end`` — the difference between stopping ON
    the last authored pose and overshooting into an extrapolation of it.
    """
    pb = Playback(end=10.0, speed=3.0)
    pb.play()
    seen = []
    for _ in range(6):
        pb.step(1)
        seen.append(pb.phase)
        pb.playing = True
    assert seen[:3] == [3.0, 6.0, 9.0], seen
    # 9 + 3 = 12 > 10, so it clamps to `end` and stops rather than showing 12.
    assert seen[3] == 10.0, seen
    assert max(seen) <= 10.0, "playback went past the clip's last keyframe"
    print("gate              phase+speed<=end, tested first: %s" % seen[:4])


def test_a_non_looping_clip_stops_itself():
    pb = Playback(end=8.0, speed=2.0, loop=False)
    pb.play()
    pb.advance(10.0)
    assert pb.phase == 8.0 and not pb.playing, pb
    print("one-shot          stops at end and clears `playing`")


def test_a_looping_clip_wraps_and_keeps_the_overshoot():
    pb = Playback(end=10.0, speed=3.0, loop=True)
    pb.play()
    for _ in range(4):
        pb._one(1.0)
    # 3, 6, 9, then 9+3=12 > 10 -> wraps by the 2 it went over, not back to 0.
    assert abs(pb.phase - 2.0) < 1e-6, pb.phase
    assert pb.playing, "a looping clip must keep playing"
    pb.advance(10.0)
    assert 0.0 <= pb.phase <= 10.0 and pb.playing, pb
    print("loop              wraps carrying the overshoot, never stops")


def test_wall_clock_is_end_over_speed_over_30():
    """The identity from the live clip-state block: `end / speed / 30`."""
    assert abs(wall_clock(154, 2.0) - 154 / 2.0 / 30.0) < 1e-12
    assert abs(wall_clock(154, 2.4) - 154 / 2.4 / 30.0) < 1e-12
    # 🔴 the same clip is a DIFFERENT number of seconds at the two observed rates —
    # which is the point: the span is the clip's, the rate is the action's.
    assert wall_clock(154, 2.0) > wall_clock(154, 2.4)
    assert wall_clock(10, 0) == float("inf"), "speed 0 must not divide by zero"
    print("wall clock        %.3f s at 2.0 vs %.3f s at 2.4 for one 154-frame clip"
          % (wall_clock(154, 2.0), wall_clock(154, 2.4)))


def test_advance_runs_at_the_engine_rate_whatever_the_display_does():
    """One second of real time is 30 game frames, at 60 fps or at 240."""
    for fps in (30.0, 60.0, 144.0, 240.0):
        pb = Playback(end=1e9, speed=1.0)
        pb.play()
        for _ in range(int(fps)):
            pb.advance(1.0 / fps)
        assert abs(pb.phase - GAME_HZ) < 1e-6, \
            "at %g fps one second advanced %.3f frames, not %g" % (fps, pb.phase, GAME_HZ)
    print("clock             1 s == %g game frames at 30/60/144/240 fps" % GAME_HZ)


def test_a_long_stall_does_not_teleport():
    pb = Playback(end=1e9, speed=2.0)
    pb.play()
    pb.advance(30.0)            # a dragged window, or a breakpoint
    assert pb.phase <= 0.25 * GAME_HZ * 2.0 + 1e-6, pb.phase
    print("clock             a 30 s stall advances at most 0.25 s of animation")


def test_seek_and_step_clamp():
    pb = Playback(end=20.0, speed=2.0)
    pb.seek(999.0)
    assert pb.phase == 20.0
    pb.seek(-5.0)
    assert pb.phase == 0.0
    pb.step(-1)
    assert pb.phase == 0.0, "stepping back from 0 on a one-shot must stay at 0"
    pb.playing = True
    pb.step(1)
    assert pb.phase == 2.0 and not pb.playing, "step must pause"
    print("transport         seek clamps, step pauses and honours the gate")


# --------------------------------------------------------------------------- #
# root motion — needs a PAC
# --------------------------------------------------------------------------- #
def test_the_travel_joints_are_not_just_the_roots():
    """🔴 On the native Tigrex the ROOT carries no location; the hip does.

    `strip_root_motion` in Blender deletes location on root bones, which on this PAC
    strips nothing at all. Issue #7 says the motion is "on the hip joint", and the
    principled set is the fork rule's leading-origin chain.
    """
    if not TIGREX.exists():
        print("SKIP: no game data (docs/ASSETS.md)")
        return
    from mhfu_monster_editor.core import open_scene

    sc = open_scene(TIGREX)
    assert root_joints(sc.rig) == (0, 45), root_joints(sc.rig)
    assert tuple(int(j) for j in travel_joints(sc)) == (0, 1, 2), travel_joints(sc)

    clip = sc.clip(3)
    curves = sc.curves(clip)
    a, b = curves.eval(0.0)[1], curves.eval(clip.frames * 0.5)[1]
    moved = np.abs(a - b).max(axis=1)
    assert moved[0] < 1e-9, "joint 0 (the root) should carry no location on this PAC"
    assert moved[1] > 100.0, "joint 1 (the hip) should carry the travel, got %.1f" % moved[1]
    print("travel joints     root 0 moves %.2f u, hip 1 moves %.0f u -> strip the "
          "lead chain (0,1,2), not the roots" % (moved[0], moved[1]))


def test_root_strip_removes_travel_and_keeps_frame_zero():
    """The two properties "plays in place" actually means."""
    for path, label in ((TIGREX, "tigrex"), (ZINOGRE, "zinogre")):
        if not path.exists():
            print("SKIP: %s is not here" % path.name)
            continue
        from mhfu_monster_editor.core import open_scene

        sc = open_scene(path)
        best = max(sc.clips, key=lambda c: root_travel(sc, c)[1])
        peak = root_travel(sc, best)[1]
        assert peak > 100.0, "%s: no clip travels far enough to test" % label

        # 1. frame 0 is IDENTICAL — the strip holds the frame-0 value, so it cannot
        #    drop the animal to its bind height and plant it in the floor.
        a = pose_at(sc, best, 0.0).joints
        b = pose_at(sc, best, 0.0, strip_root=True).joints
        assert np.allclose(a, b, atol=1e-12), "%s: frame 0 changed under the strip" % label

        # 2. and the subject stops leaving the frame.
        frames = np.linspace(0.0, best.frames, 20)
        def drift(strip):
            c = np.array([pose_at(sc, best, f, strip_root=strip).joints.mean(0)
                          for f in frames])
            return float(np.linalg.norm(c - c[0], axis=1).max())
        kept, stripped = drift(False), drift(True)
        assert stripped < kept * 0.25, \
            "%s: strip only cut drift from %.0f to %.0f" % (label, kept, stripped)
        print("root strip        %-8s slot %2d travels %.0f u; centroid drift "
              "%.0f -> %.0f u, frame 0 unchanged" % (label, best.slot, peak, kept, stripped))


def test_root_travel_separates_a_lunge_from_a_stand():
    if not TIGREX.exists():
        print("SKIP: no game data")
        return
    from mhfu_monster_editor.core import open_scene

    sc = open_scene(TIGREX)
    travels = {c.slot: root_travel(sc, c)[1] for c in sc.clips}
    movers = [s for s, t in travels.items() if t > 100.0]
    still = [s for s, t in travels.items() if t < 1.0]
    assert movers and still, \
        "travel does not discriminate: %d movers, %d still" % (len(movers), len(still))
    print("root travel       %d clips travel >100 u, %d stay in place (max %.0f)"
          % (len(movers), len(still), max(travels.values())))


# --------------------------------------------------------------------------- #
# GL — issue #7's "done when"
# --------------------------------------------------------------------------- #
def _play_every_clip(ctx, path, label):
    """#7: 'the native Tigrex and the Zinogre port both play every clip cleanly'."""
    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.render.viewport import Viewport

    sc = open_scene(path)
    n_frames = 0
    t0 = time.perf_counter()
    with Viewport(ctx, (480, 360)) as vp:
        vp.set_scene(sc)
        bg = np.array([int(round(c * 255)) for c in vp.background[:3]])
        for c in sc.clips:
            vp.play_clip(c)
            assert vp.playback.end == c.frames, (c.slot, vp.playback.end, c.frames)
            assert vp.playback.loop == c.loop, c.slot
            for f in np.linspace(0.0, c.frames, 7):
                vp.set_pose(c, float(f))
                joints = vp.skeleton.positions
                assert np.all(np.isfinite(joints)), \
                    "%s slot %d frame %.0f produced a non-finite joint" % (label, c.slot, f)
                assert np.abs(joints).max() < 1e7, \
                    "%s slot %d frame %.0f flung a joint to %.3g — the clip is not " \
                    "being read as this rig's" % (label, c.slot, f, np.abs(joints).max())
                vp.draw()
                n_frames += 1
            img = vp.target.read()
            lit = int((np.abs(img[..., :3].astype(int) - bg).max(axis=2) > 12).sum())
            assert lit > 200, "%s slot %d drew only %d pixels" % (label, c.slot, lit)
    dt = time.perf_counter() - t0
    return len(sc.clips), n_frames, dt


def test_every_clip_plays(ctx):
    done = False
    for path, label in ((TIGREX, "tigrex"), (ZINOGRE, "zinogre")):
        if not path.exists():
            print("SKIP: %s is not here (game data — docs/ASSETS.md)" % path.name)
            continue
        clips, frames, dt = _play_every_clip(ctx, path, label)
        rate = frames / dt if dt else 0.0
        assert rate > 20.0, "%s rendered at %.0f fps — not interactive" % (label, rate)
        print("play all          %-8s %2d clips, %3d posed+drawn frames in %.2f s "
              "(%.0f fps)" % (label, clips, frames, dt, rate))
        done = True
    if not done:
        print("SKIP: neither PAC is available")


def test_the_transport_drives_the_viewport(ctx):
    if not TIGREX.exists():
        print("SKIP: no game data")
        return
    from mhfu_monster_editor.core import open_scene
    from mhfu_monster_editor.render.viewport import Viewport

    sc = open_scene(TIGREX)
    with Viewport(ctx, (200, 160)) as vp:
        vp.set_scene(sc)
        clip = next(c for c in sc.clips if c.loop and c.frames > 40)
        vp.play_clip(clip, 0)
        assert not vp.tick(1 / 60.0), "a paused transport must not re-pose"
        vp.playback.play()
        # at 60 fps a display frame is HALF a game frame, so the first tick advances
        # nothing and banks the carry — that is the engine's 30 Hz, not a bug. The
        # second tick spends it.
        assert not vp.tick(1 / 60.0), "half a game frame should not advance the clip"
        assert vp.tick(1 / 60.0), "the carried half-frame should complete on the next tick"
        assert vp.playback.phase > 0 and vp.frame == vp.playback.phase, vp.playback
        # and the mesh really follows: a second of playback must change the picture.
        vp.draw()
        a = vp.target.read()
        for _ in range(30):
            vp.tick(1 / 30.0)
        vp.draw()
        assert not np.array_equal(a, vp.target.read()), "playback did not move the mesh"
    print("transport         paused -> no re-pose; playing -> the mesh moves")


def main() -> int:
    for fn in (test_the_gate_is_tested_before_the_step,
               test_a_non_looping_clip_stops_itself,
               test_a_looping_clip_wraps_and_keeps_the_overshoot,
               test_wall_clock_is_end_over_speed_over_30,
               test_advance_runs_at_the_engine_rate_whatever_the_display_does,
               test_a_long_stall_does_not_teleport,
               test_seek_and_step_clamp,
               test_the_travel_joints_are_not_just_the_roots,
               test_root_strip_removes_travel_and_keeps_frame_zero,
               test_root_travel_separates_a_lunge_from_a_stand):
        fn()

    try:
        from mhfu_monster_editor.render.context import ContextError, describe, headless
    except ImportError as e:
        print("SKIP: moderngl is not installed (%s)" % e)
        print("\ntest_render_playback: OK (no-GL part only)")
        return 0
    try:
        ctx = headless()
    except ContextError as e:
        print("SKIP: no GL on this machine.\n%s" % e)
        print("\ntest_render_playback: OK (no-GL part only)")
        return 0
    print("context           %s" % describe(ctx))
    try:
        test_every_clip_plays(ctx)
        test_the_transport_drives_the_viewport(ctx)
    finally:
        ctx.release()
    print("\ntest_render_playback: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
