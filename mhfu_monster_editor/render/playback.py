"""Playing a clip the way the ENGINE plays it — and no GL, so it is testable anywhere.

This is not "a float that goes up". The clip-state block the engine keeps per body-part
slot was read live off a running game (`docs/agent_memory_map.md`, entity
``+0x80 + slot*0x40``, confirmed on the ported Zinogre over ~2000 samples), and it says
exactly how a clip advances:

===============  ======  ====================================================
field            offset  meaning
===============  ======  ====================================================
``phase``        +0x10   the playback cursor, **in clip frames**
``speed``        +0x14   frames advanced per GAME frame
``loop_restart`` +0x18   where a looping clip resumes
``end``          +0x1C   the clip's own last keyframe
flags            +0x3C   bit0 playing, bit1 loop  (mirrored at ``+0xBC``)
===============  ======  ====================================================

Three consequences this module implements literally:

1. 🔴 **The rate is the ACTION's, the span is the CLIP's.** ``speed`` comes from the
   action dispatch, not from the clip — **2.0 and 2.4 were both observed on the same
   monster**. So a clip has no wall-clock duration of its own; it has a length in
   frames, and what it is dispatched at decides the seconds:
   :func:`wall_clock` = ``end / speed / 30``.
2. 🔴 **There is no per-action duration field anywhere.** The engine plays to the
   clip's own last keyframe and then loops iff the clip's loop flag is set
   (`docs/ANIMATION_FORMAT.md`; the Tigrex action-descriptor table has no frame count).
   That is why a clip padded to a fixed 180 frames FREEZES on its last pose until 180 —
   it is not a stall, it is the pad being played.
3. **The advance gate is** ``phase + speed <= end`` (the compare at ``0x088638E0``),
   which is tested BEFORE the step. A clip therefore never shows a frame past ``end``.

:data:`GAME_HZ` is 30: the PSP game frame. Everything here is in clip frames and game
frames, never in seconds, until :func:`wall_clock` is asked.
"""
from __future__ import annotations

from typing import Optional, Sequence, Tuple

import numpy as np

#: the engine's tick. `wall_clock` is the only place seconds appear.
GAME_HZ = 30.0

#: `speed` when nothing says otherwise. 2.0 and 2.4 were both measured live; 2.0 is
#: the one to default to and the UI exposes it, because it is a property of the ACTION
#: that plays the clip and this tool is not running an action.
DEFAULT_SPEED = 2.0

#: the two rates actually seen in a live capture, offered as presets.
OBSERVED_SPEEDS = (1.0, 2.0, 2.4)


def wall_clock(end: float, speed: float = DEFAULT_SPEED) -> float:
    """Seconds a clip of ``end`` frames takes at ``speed`` — ``end / speed / 30``."""
    if speed <= 0:
        return float("inf")
    return float(end) / float(speed) / GAME_HZ


class Playback:
    """The engine's clip cursor: ``phase``, ``speed``, and the ``phase + speed <= end``
    gate. Frame-accurate rather than time-accurate, then reconciled to wall time.

    :meth:`advance` takes REAL seconds and converts, so playback runs at the engine's
    30 Hz whatever the window renders at — a 120 Hz display must not play a clip four
    times too fast, and a stuttering frame must not skip the animation. The leftover
    fraction of a game frame is carried, so a long run drifts by nothing.
    """

    def __init__(self, end: float = 0.0, *, loop: bool = False,
                 speed: float = DEFAULT_SPEED, loop_restart: float = 0.0) -> None:
        self.end = float(end)
        self.loop = bool(loop)
        self.speed = float(speed)
        self.loop_restart = float(loop_restart)
        self.phase = 0.0
        self.playing = False
        #: fraction of a game frame carried between :meth:`advance` calls.
        self._carry = 0.0

    # ---- binding ------------------------------------------------------ #
    def set_clip(self, clip, *, keep_phase: bool = False) -> "Playback":
        """Adopt a :class:`~mhfu_monster_editor.core.scene.Clip`'s span and loop flag."""
        self.end = float(getattr(clip, "frames", 0) or 0)
        self.loop = bool(getattr(clip, "loop", False))
        self.phase = min(self.phase, self.end) if keep_phase else 0.0
        self._carry = 0.0
        return self

    # ---- transport ---------------------------------------------------- #
    def play(self) -> None:
        self.playing = True

    def pause(self) -> None:
        self.playing = False

    def toggle(self) -> None:
        self.playing = not self.playing

    def seek(self, frame: float) -> None:
        """Jump to a frame, clamped into ``0..end``. Scrubbing does not stop playback."""
        self.phase = max(0.0, min(float(frame), self.end))
        self._carry = 0.0

    def step(self, game_frames: int = 1) -> None:
        """Nudge by whole GAME frames — i.e. by ``speed`` clip frames each. Pauses.

        Stepping is for reading a pose, so it stops playback the way a transport's
        step button does, and it honours the same gate as :meth:`advance` so a stepped
        frame is one the engine could actually show.
        """
        self.playing = False
        for _ in range(abs(int(game_frames))):
            self._one(-1.0 if game_frames < 0 else 1.0)

    def rewind(self) -> None:
        self.phase = 0.0
        self._carry = 0.0

    # ---- the clock ---------------------------------------------------- #
    def advance(self, dt: float) -> None:
        """Run for ``dt`` REAL seconds at the engine's 30 Hz. No-op when paused.

        ``dt`` is clamped: a window that was dragged, or a breakpoint, produces a
        multi-second delta, and replaying two hundred game frames at once turns a
        pause into a teleport.
        """
        if not self.playing or self.end <= 0:
            return
        dt = max(0.0, min(float(dt), 0.25))
        want = dt * GAME_HZ + self._carry
        # ⚠️ the epsilon is not decoration. At 144 fps a game frame is 0.2083…
        # repeating; summing it 144 times lands on 29.999999999999996, and a bare
        # int() then drops the last frame of every second — a real, if small, drift
        # against the engine that only shows on a display whose rate is not a factor
        # of 30.
        whole = int(want + 1e-9)
        self._carry = max(0.0, want - whole)
        for _ in range(whole):
            if not self.playing:
                break
            self._one(1.0)

    def _one(self, direction: float) -> None:
        """One game frame. THE gate: ``phase + speed <= end``, tested before stepping."""
        nxt = self.phase + self.speed * direction
        if direction > 0:
            if nxt <= self.end:
                self.phase = nxt
                return
            if self.loop:
                # the engine resumes at the clip's loop-restart, carrying the overshoot
                # so a loop does not lose the fraction of a frame it went past by.
                span = max(self.end - self.loop_restart, 1e-6)
                self.phase = self.loop_restart + (nxt - self.end - 1e-9) % span
            else:
                self.phase = self.end
                self.playing = False
        else:
            if nxt >= 0.0:
                self.phase = nxt
            elif self.loop:
                span = max(self.end - self.loop_restart, 1e-6)
                self.phase = self.end - (-nxt) % span
            else:
                self.phase = 0.0

    # ---- reporting ---------------------------------------------------- #
    @property
    def progress(self) -> float:
        return 0.0 if self.end <= 0 else self.phase / self.end

    @property
    def duration(self) -> float:
        """Wall-clock seconds for the whole clip at the current ``speed``."""
        return wall_clock(self.end, self.speed)

    def __repr__(self) -> str:
        return ("<Playback %.1f/%.0f speed=%.2f %s%s>"
                % (self.phase, self.end, self.speed,
                   "playing" if self.playing else "paused",
                   " loop" if self.loop else ""))


# --------------------------------------------------------------------------- #
# root motion
# --------------------------------------------------------------------------- #
def root_joints(rig) -> Tuple[int, ...]:
    """Joints with no parent. A big-monster PAC can have more than one."""
    return tuple(int(i) for i, p in enumerate(np.asarray(rig.parents)) if p < 0)


def travel_joints(scene) -> Tuple[int, ...]:
    """The joints whose LOCATION moves the whole animal — the leading-origin chain.

    🔴 **Not just the roots.** `blender_mhfu/render_anim_clips.strip_root_motion`
    deletes location on root bones, and on the native Tigrex that strips *nothing*: its
    root (joint 0) carries no location channel at all, and the 357 units of travel sit
    on **joint 1**. Issue #7 says as much — "clips carry root motion on the hip joint".

    The principled set is the one `p3rd_anim_map`'s fork rule already defines: the body
    fork and every ancestor of it. A location channel translates its joint's whole
    subtree, so those are exactly the joints that can move the ENTIRE animal; one below
    the fork moves half of it and is articulation, not travel. Falls back to the roots
    for a rig with no fork.
    """
    from .skeleton import leading_chain
    chain = leading_chain(scene.rig.parents)
    return chain or root_joints(scene.rig)


def pose_at(scene, clip, frame: float, *, strip_root: bool = False):
    """The rig at ``frame``, optionally with the clip's travel removed.

    ``strip_root`` holds the location channels of :func:`travel_joints` at their
    **frame-0 values**, so the clip plays in place and a locomotion cycle stays in
    frame instead of walking out of it. Rotation is kept throughout, as the Blender
    oracle keeps it.

    ⚠️ Frame 0, **not** the bind translation, and that is the one real departure from
    `strip_root_motion`. Deleting the channel drops the joint to its bind position, and
    on a monster whose standing height comes from that very channel — the Tigrex is
    lifted ~357 units by joint 1 — that plants the animal in the floor. Holding the
    frame-0 value removes the drift and nothing else: at frame 0 the stripped and
    unstripped poses are identical, and every later frame differs from the unstripped
    one by the travel alone.

    ⚠️ Issue #7 also warns to strip *after* the pose is built, because in Blender
    `bake_all` deletes and rebuilds every Action and so silently undoes an earlier
    strip. That trap cannot recur here: there is no baked state to undo. The pose is
    rebuilt from the channels on every call, and the substitution happens between
    sampling them and running the FK — which is what deleting an fcurve does.
    """
    if clip is None:
        return scene.bind_pose()
    c = scene.clip(clip)
    curves = scene.curves(c)
    rot, loc = curves.eval(float(frame))
    if strip_root:
        js = list(travel_joints(scene))
        if js:
            loc = np.array(loc, copy=True)
            loc[js] = curves.eval(0.0)[1][js]
    from ..core.pose import Pose
    return Pose(rig=scene.rig, frame=float(frame),
                world=scene.rig.world(rot, loc), slot=c.slot)


def root_travel(scene, clip) -> Tuple[float, float]:
    """``(net, peak)`` distance the animal travels over the clip, in engine units.

    Worth having in the clip list rather than only in the viewport, because stripping
    the travel is what makes a forward lunge and a standing bite look alike — the
    travel IS half of what the move is. `render_anim_clips.root_travel` puts the same
    number in its index: 0 = in place, ~1000 = about one body length on a Tigrex.

    Measured over :func:`travel_joints`, for the same reason that function exists: on
    the native Tigrex the root moves not at all and the hip moves 357 units.
    """
    c = scene.clip(clip)
    roots = travel_joints(scene)
    if not roots or c.frames <= 0:
        return 0.0, 0.0
    curves = scene.curves(c)
    # one sample per GAME frame at the default rate — enough to catch the peak of a
    # lunge without evaluating 520 frames of a long ambient loop.
    steps = max(2, min(int(c.frames / DEFAULT_SPEED) + 1, 200))
    path = np.array([curves.eval(f)[1][list(roots)].sum(axis=0)
                     for f in np.linspace(0.0, c.frames, steps)])
    d = np.linalg.norm(path - path[0], axis=1)
    return float(np.linalg.norm(path[-1] - path[0])), float(d.max())
