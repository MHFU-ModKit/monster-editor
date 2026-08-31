"""The bone overlay: joints, the bones between them, and the two joints that matter.

Issue #6 asks for "the leading-origin chain and the body **fork** joint highlighted
(the fork rule is what the whole anim record→bone mapping turns on)". Both come from
`mhfu_model.p3rd_anim_map`, read rather than re-derived:

* **the fork** — :func:`p3rd_anim_map.body_fork`, the first joint with more than one
  child, where the rig splits into front and back. On the native Tigrex that is joint 2
  (children 3, 21, 26, 40).
* **the leading-origin chain** — the fork and everything above it, which is exactly the
  set of joints a LOCATION channel may legitimately land on. A location channel
  translates its joint's whole subtree, so one below the fork lifts half the animal and
  tears the waist. :func:`p3rd_anim_map.loc_below_fork` is the check; this draws the
  region that check is about, so a bad record→bone map is visible instead of inferred.

Colour carries meaning here, so it is worth stating: **amber = the fork**, **green =
the leading-origin chain**, **red = whatever you selected**, grey-blue for the rest.
"""
from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import numpy as np

from ..core._paths import tools_on_path

tools_on_path()

from mhfu_model import p3rd_anim_map as P3AM       # noqa: E402

#: joint colours, RGBA.
C_BONE = (0.42, 0.52, 0.66, 0.90)
C_JOINT = (0.62, 0.72, 0.86, 1.00)
C_FORK = (0.98, 0.70, 0.20, 1.00)
C_LEAD = (0.36, 0.82, 0.42, 1.00)
C_SELECTED = (0.95, 0.20, 0.22, 1.00)
C_DRIVEN = (0.55, 0.62, 0.78, 1.00)
C_UNDRIVEN = (0.40, 0.40, 0.44, 0.75)


def leading_chain(parents: Sequence[int]) -> Tuple[int, ...]:
    """The fork and every ancestor of it — where a location channel is allowed.

    Returns an empty tuple for a rig with no fork at all (a single chain), which is not
    an error: there is simply nothing to divide.
    """
    par = list(parents)
    fork = P3AM.body_fork(par)
    if fork < 0:
        return ()
    chain = [fork]
    seen = {fork}
    j = par[fork] if fork < len(par) else -1
    while 0 <= j < len(par) and j not in seen:
        chain.append(j)
        seen.add(j)
        j = par[j]
    return tuple(reversed(chain))


def body_fork(parents: Sequence[int]) -> int:
    """`p3rd_anim_map.body_fork`, re-exported so callers need not reach into `tools/`."""
    return P3AM.body_fork(list(parents))


def undriven_geometry(scene) -> "dict[int, int]":
    """``{joint: vertex count}`` for joints that carry geometry **no clip drives**.

    Worth surfacing rather than merely drawing, because the picture alone is baffling:
    that geometry sits at its BIND position — usually the world origin — so it renders
    as a detached lump beside the animal and reads as a skinning bug.

    It is not one. The **native Tigrex** (`file_06185`) has a skeleton of 48 joints
    while every clip's track map stops at 44: joints 45 -> 46 -> 47 are a SECOND root
    chain, and 150 vertices are dominantly weighted to 46 and 47. The animation
    streams partition 31/9/5 = 45 joints, so the pack simply has nothing to say about
    the last three; whatever places them in game is not a clip in this PAC.

    The Zinogre port inherits the same shape, which is the useful part: a port that
    accidentally weights real body geometry onto an undriven joint would show up here
    as a large count instead of a small one.
    """
    if not scene.clips:
        return {}
    ever = set()
    for c in scene.clips:
        ever |= set(c.driven)
    dom = scene.merged.dominant()
    out = {}
    for j in range(scene.rig.n_bones):
        if j in ever:
            continue
        n = int((dom == j).sum())
        if n:
            out[j] = n
    return out


class SkeletonOverlay:
    """Joints as points and bones as segments, coloured by role and selection."""

    def __init__(self, ctx, scene) -> None:
        from .overlay import Lines

        self.ctx = ctx
        self.scene = scene
        self.parents = np.asarray(scene.rig.parents, dtype=np.int32)
        self.fork = body_fork(self.parents)
        self.lead = leading_chain(self.parents)
        self.selected: Optional[int] = None
        #: when set, joints this clip does not drive are drawn dimmed — the fastest
        #: read on "is this clip actually moving the part I think it is?".
        self.driven: Optional[Tuple[int, ...]] = None
        self.joint_size = 6.0

        self._bones = Lines(ctx)
        self._joints = Lines(ctx, mode=ctx.POINTS)
        self._pairs = np.array([(int(p), i) for i, p in enumerate(self.parents)
                                if 0 <= p < len(self.parents)], dtype=np.int32)
        self._positions = scene.rig.bind_joints.copy()
        self.set_positions(self._positions)

    # ---- state -------------------------------------------------------- #
    def set_positions(self, joints: np.ndarray) -> None:
        """Move the overlay onto a pose. ``joints`` is ``(n,3)`` world positions."""
        self._positions = np.asarray(joints, dtype=np.float64).reshape(-1, 3)
        self._rebuild()

    def set_selected(self, joint: Optional[int]) -> None:
        if joint == self.selected:
            return
        self.selected = joint
        self._rebuild()

    def set_driven(self, driven) -> None:
        d = tuple(sorted(driven)) if driven is not None else None
        if d == self.driven:
            return
        self.driven = d
        self._rebuild()

    @property
    def positions(self) -> np.ndarray:
        return self._positions

    # ---- colouring ---------------------------------------------------- #
    def joint_colors(self) -> np.ndarray:
        n = len(self._positions)
        col = np.tile(np.array(C_JOINT, dtype="f4"), (n, 1))
        if self.driven is not None:
            col[:] = np.array(C_UNDRIVEN, dtype="f4")
            idx = [j for j in self.driven if 0 <= j < n]
            if idx:
                col[idx] = np.array(C_DRIVEN, dtype="f4")
        for j in self.lead:
            if 0 <= j < n:
                col[j] = np.array(C_LEAD, dtype="f4")
        if 0 <= self.fork < n:
            col[self.fork] = np.array(C_FORK, dtype="f4")
        if self.selected is not None and 0 <= self.selected < n:
            col[self.selected] = np.array(C_SELECTED, dtype="f4")
        return col

    def _rebuild(self) -> None:
        p = self._positions
        if len(self._pairs):
            seg = p[self._pairs.reshape(-1)]
            jc = self.joint_colors()
            # a bone takes its child's colour, so the fork's own bone reads as the fork.
            cols = np.repeat(jc[self._pairs[:, 1]], 2, axis=0) * 0.85
            cols[:, 3] = C_BONE[3]
            self._bones.set(seg, cols)
        self._joints.set(p, self.joint_colors())

    # ---- drawing ------------------------------------------------------ #
    def render(self, mvp: np.ndarray, *, alpha: float = 1.0) -> None:
        self._bones.alpha = alpha
        self._joints.alpha = alpha
        self._joints.point_size = self.joint_size
        self._bones.render(mvp)
        self._joints.render(mvp)

    # ---- picking ------------------------------------------------------ #
    def pick(self, mvp: np.ndarray, size: Tuple[int, int], x: float, y: float,
             radius: float = 14.0) -> Optional[int]:
        """The joint nearest ``(x, y)`` in PANEL pixels, or None past ``radius``.

        ``y`` is measured from the TOP of the panel — imgui's convention, not GL's.
        Joints behind the camera are excluded rather than wrapped around by the
        perspective divide, which would otherwise let a tail joint be picked from in
        front of the face.
        """
        pts = self.project(mvp, size)
        if pts is None:
            return None
        xy, ok = pts
        if not ok.any():
            return None
        d = np.hypot(xy[:, 0] - x, xy[:, 1] - y)
        d[~ok] = np.inf
        j = int(np.argmin(d))
        return j if d[j] <= radius else None

    def project(self, mvp: np.ndarray, size: Tuple[int, int]):
        """``((n,2) panel pixels, (n,) visible)``. y from the TOP. None if empty."""
        p = self._positions
        if not len(p):
            return None
        clip = np.concatenate([p, np.ones((len(p), 1))], axis=1) @ np.asarray(mvp).T
        w = clip[:, 3]
        ok = w > 1e-9
        ndc = np.zeros((len(p), 3))
        ndc[ok] = clip[ok, :3] / w[ok, None]
        xy = np.empty((len(p), 2))
        xy[:, 0] = (ndc[:, 0] * 0.5 + 0.5) * size[0]
        xy[:, 1] = (1.0 - (ndc[:, 1] * 0.5 + 0.5)) * size[1]
        ok &= np.all(np.abs(ndc[:, :2]) <= 1.2, axis=1)
        return xy, ok

    def release(self) -> None:
        self._bones.release()
        self._joints.release()

    def __repr__(self) -> str:
        return ("<SkeletonOverlay %d joints, fork=%d, lead=%s, selected=%s>"
                % (len(self._positions), self.fork, list(self.lead), self.selected))
